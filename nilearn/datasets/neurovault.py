import os
import logging
from copy import deepcopy
import shutil
import re
import json
from glob import glob
from tempfile import mkdtemp
from pprint import pprint

from .._utils.compat import _urllib
urljoin, urlencode = _urllib.parse.urljoin, _urllib.parse.urlencode
Request, build_opener = _urllib.request.Request, _urllib.request.build_opener
from .utils import _fetch_file, _get_dataset_dir


_COL_FILTERS_AVAILABLE_ON_SERVER = {'DOI', 'name', 'owner'}
_IM_FILTERS_AVAILABLE_ON_SERVER = set()


def prepare_logging(level=logging.DEBUG):
    """Get the root logger and add a handler to it if it doesn't have any.

    Parameters
    ----------
    level: int, optional (default=logging.DEBUG)
    level of the handler that is added if none exist.
    this handler streams output to the console.

    Returns
    -------
    logger: logging.RootLogger
    the root logger.

    """
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    console_logger = logging.StreamHandler()
    console_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    console_logger.setFormatter(formatter)
    logger.addHandler(console_logger)
    return logger


_logger = prepare_logging()


def _neurovault_base_url():
    return 'http://neurovault.org/api/'


def _neurovault_collections_url():
    return urljoin(_neurovault_base_url(), 'collections/')


def _neurovault_images_url():
    return urljoin(_neurovault_base_url(), 'images/')


def _append_filters_to_query(query, filters):
    """encode dict or sequence of key-value pairs into an URL query string"""
    if not filters:
        return query
    new_query = urljoin(
        query, urlencode(filters))
    return new_query


def _default_batch_size():
    return 100


def _empty_filter(arg):
    return True


def _get_batch(query, prefix_msg=''):
    request = Request(query)
    opener = build_opener()
    _logger.debug('{}getting new batch: {}'.format(
        prefix_msg, new_query))
    try:
        with opener.open(request) as resp:
            content_type = resp.getheader('Content-Type')
            content = resp.read()
    except Exception as e:
        _logger.exception(
            'could not download batch from {}'.format(query))
        return None
    try:
        encoding = re.search(r'charset=\b(.+)\b', content_type).group(1)
        batch = json.loads(content.decode(encoding))
    except Exception as e:
        _logger.exception('could not decypher batch from {}'.format(query))
        return None
    for key in ['results', 'count']:
        if batch.get(key) is None:
            _logger.error('could not find required key "{}" '
                          'in batch retrieved from {}'.format(key, query))
        return None

    return batch


def _scroll_server_results(url, local_filter=_empty_filter,
                           query_terms=None, max_results=None,
                           batch_size=None, prefix_msg=''):
    """download list of metadata from Neurovault"""
    query = _append_filters_to_query(url, query_terms)
    if batch_size is None:
        batch_size = _default_batch_size()
    query = '{}?limit={}&offset={{}}'.format(query, batch_size)
    downloaded = 0
    n_available = None
    while(max_results is None or downloaded < max_results):
        new_query = query.format(downloaded)
        batch = _get_batch(new_query, prefix_msg)
        batch_size = len(batch['results'])
        downloaded += batch_size
        _logger.debug('{}batch size: {}'.format(prefix_msg, batch_size))
        if n_available is None:
            n_available = batch['count']
            max_results = (n_available if max_results is None
                           else min(max_results, n_available))
        for result in batch['results']:
            if local_filter(result):
                yield result


class NotNull(object):

    def __eq__(self, other):
        return bool(other)

    def __req__(self, other):
        return self.__eq__(other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __rneq__(self, other):
        return self.__neq__(other)


class NotEqual(object):

    def __init__(self, negated):
        self.negated_ = negated

    def __eq__(self, other):
        return self.negated_ != other

    def __req__(self, other):
        return self.__eq__(other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __rneq__(self, other):
        return self.__neq__(other)


class ResultFilter(object):

    not_null = NotNull()

    def __init__(self, query_terms={}, callable_filter=_empty_filter, **kwargs):
        query_terms = dict(query_terms, **kwargs)
        self.query_terms_ = query_terms
        self.callable_filters_ = [callable_filter]

    def __call__(self, candidate):
        for key, value in self.query_terms_.items():
            if candidate.get(key) != value:
                return False
        for callable_filter in self.callable_filters_:
            if not callable_filter(candidate):
                return False
        return True

    def __or__(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) or filt2(r))
        return new_filter

    def __ror__(self, other_filter):
        return self.__or__(other_filter)

    def __and__(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) and filt2(r))
        return new_filter

    def __rand__(self, other_filter):
        return self.__and__(other_filter)

    def __xor__(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) ^ filt2(r))
        return new_filter

    def __rxor__(self, other_filter):
        return self.__xor__(other_filter)

    def __not__(self):
        filt = deepcopy(self)
        new_filter = ResultFilter(
            callable_filter=lambda r: not filt(r))
        return new_filter

    def __getitem__(self, ):
        return self.query_terms_[item]

    def __setitem__(self, item, value):
        self.query_terms_[item] = value

    def __delitem__(self, item):
        if item in self.query_terms_:
            del self.query_terms_[item]

    def add_filter(callable_filter):
        self.callable_filters_.append(callable_filter)


class BaseDownloadManager(object):

    def __init__(self, max_images=100):
        if max_images is not None and max_images < 0:
            max_images = None
        self.max_images_ = max_images
        self.already_downloaded_ = 0

    def collection(self, collection_info):
        return collection_info

    def image(self, image_info):
        if self.already_downloaded_ == self.max_images_:
            raise StopIteration()
        self.already_downloaded_ += 1
        return image_info


class DownloadManager(BaseDownloadManager):

    def __init__(self, data_dir=None, temp_dir=None, **kwargs):
        super(DownloadManager, self).__init__(**kwargs)
        if data_dir is None:
            data_dir = _get_dataset_dir('neurovault')
        self.data_dir_ = os.path.abspath(os.path.expanduser(data_dir))
        if not os.path.isdir(self.data_dir_):
            os.makedirs(self.data_dir_)
        if not os.path.isdir(temp_dir):
            temp_dir = mkdtemp()
        self.temp_dir_ = temp_dir

    def collection(self, collection_info):
        collection_id = collection_info['id']
        collection_dir = os.path.join(self.data_dir_, collection_id)
        collection_info['local_path'] = collection_dir
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(collection_dir,
                                          'collection_metadata.json')
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(collection_info, metadata_file)
        return collection_info

    def image(self, image_info):
        collection_id = image_info['collection_id']
        collection_dir = os.path.join(self.data_dir_, collection_id)
        image_id = image_info['id']
        image_url = image_info['url']
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(
            collection_dir, 'image_{}_metadata.json'.format(image_id))
        image_file_path = os.path.join(
            collection_dir, 'image_{}__.nii.gz'.format(image_id))
        downloaded = _fetch_file(image_url, self.temp_dir_, resume=False,
                                 overwrite=True, verbose=0)
        shutil.move(downloaded, image_file_path)
        image_info['local_path'] = image_file_path
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(image_info, metadata_file)

        return image_info


def _scroll_server_data(collection_query_terms={},
                        collection_local_filter=_empty_filter,
                        image_query_terms={},
                        image_local_filter=_empty_filter,
                        download_manager=None, max_images=None,
                        metadata_batch_size=None):
    if download_manager is None:
        download_manager = BaseDownloadManager(max_images=max_images)

    collections = _scroll_server_results(_neurovault_collections_url(),
                                         query_terms=collection_query_terms,
                                         local_filter=collection_local_filter,
                                         prefix_msg='scroll collections: ',
                                         batch_size=metadata_batch_size)
    for collection in collections:
        bad_collection = False
        try:
            collection = download_manager.collection(collection)
        except Exception as e:
            _logger.exception('_scroll_server_data: bad collection: {}'.format(
                collection))
            bad_collection = True

        if not bad_collection:
            query = urljoin(_neurovault_collections_url(),
                            '{}/images/'.format(collection['id']))
            images = _scroll_server_results(query,
                                            query_terms=image_query_terms,
                                            local_filter=image_local_filter,
                                            prefix_msg='scroll images: ',
                                            batch_size=metadata_batch_size)
            for image in images:
                try:
                    image = download_manager.image(image)
                    yield image, collection
                except Exception as e:
                    _logger.exception(
                        '_scroll_server_data: bad image: {}'.format(image))


def _json_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as dumped:
        loaded = json.load(dumped)
    return loaded


def _json_add_local_dir(filename):
    loaded = _json_from_file(filename)
    loaded.setdefault('local_path', os.path.dirname(filename))
    return loaded


def _json_add_local_path(filename):
    loaded = _json_from_file(filename)
    loaded.setdefault('local_path', filename)
    return loaded


def _scroll_local_data(neurovault_dir,
                       collection_filter=_empty_filter,
                       image_filter=_empty_filter,
                       max_images=None):
    if max_images is not None and max_images < 0:
        max_images = None
    found_images = 0
    neurovault_dir = os.path.abspath(os.path.expanduser(neurovault_dir))
    collections = glob(
        os.path.join(neurovault_dir, '*', 'collection_metadata.json'))

    for collection in filter(collection_filter,
                             map (_json_add_local_dir, collections)):
        images = glob(os.path.join(
            collection['local_path'], 'image_*_metadata.json'))
        # for compatibility with previous PR
        images.extend(glob(os.path.join(
            collection['local_path'], '*nii_metadata.json')))
        for image in filter(image_filter, map(_json_add_local_path, images)):
            if found_images == max_images:
                return
            found_images += 1
            yield image, collection


def _split_terms(terms, available_on_server):
    """Isolate term filters that can be applied by server."""
    terms_ = dict(terms)
    server_terms = {k: terms_.pop(k) for
                      k in available_on_server.intersection(terms_.keys())}
    return terms_, server_terms


def _move_unknown_terms_to_local_filter(terms, local_filter,
                                            available_on_server):

    local_terms, server_terms = _split_terms(terms, available_on_server)
    local_filter = local_filter & ResultFilter(query_terms=local_terms)
    return server_terms, local_filter


def _prepare_local_scroller(neurovault_dir, collection_terms, collection_filter,
                            image_terms, image_filter, max_images):

    collection_local_filter = (collection_filter
                               & ResultFilter(**collection_terms))
    image_local_filter = (image_filter
                          & ResultFilter(**image_terms))
    local_data = _scroll_local_data(
        neurovault_dir, collection_filter=collection_local_filter,
        image_filter=image_local_filter, max_images=max_images)

    return local_data


def _prepare_remote_scroller(collection_terms, collection_filter,
                             image_terms, image_filter,
                             collection_ids, image_ids,
                             download_manager, max_images):

    collection_terms, collection_filter = _move_unknown_terms_to_local_filter(
        collection_terms, collection_filter,
        _COL_FILTERS_AVAILABLE_ON_SERVER)

    collection_filter = collection_filter & ResultFilter(
        callable_filter=lambda c: c['id'] not in collection_ids)

    image_terms, image_filter = _move_unknown_terms_to_local_filter(
        image_terms, image_filter,
        _IM_FILTERS_AVAILABLE_ON_SERVER)

    image_filter = image_filter & ResultFilter(
        callable_filter=lambda i: i['id'] not in image_ids)

    if download_manager is not None:
        download_manager.already_downloaded_ = len(image_ids)

    if max_images is not None:
        max_images = max(0, max_images - len(image_ids))
    server_data = _scroll_server_data(collection_query_terms=collection_terms,
                                      collection_local_filter=collection_filter,
                                      image_query_terms=image_terms,
                                      image_local_filter=image_filter,
                                      download_manager=download_manager,
                                      max_images=max_images)
    return server_data


def _join_local_and_remote(neurovault_dir, mode='download_new',
                           collection_terms={}, collection_filter=_empty_filter,
                           image_terms={}, image_filter=_empty_filter,
                           download_manager=None, max_images=None):

    if mode not in ['overwrite', 'download_new', 'offline']:
        raise ValueError(
            'supported modes are overwrite,'
            ' download_new, offline, got {}'.format(mode))

    if mode == 'overwrite':
        local_data = tuple()
    else:
        local_data = _prepare_local_scroller(
            neurovault_dir, collection_terms, collection_filter,
            image_terms, image_filter, max_images)
    image_ids, collection_ids = set(), set()

    for image, collection in local_data:
        image_ids.add(image['id'])
        collection_ids.add(collection['id'])
        yield image, collection

    if  mode == 'offline':
        return
    if max_images is not None and len(image_ids) >= max_images :
        return

    server_data = _prepare_remote_scroller(collection_terms, collection_filter,
                                           image_terms, image_filter,
                                           collection_ids, image_ids,
                                           download_manager, max_images)
    for image, collection in server_data:
        yield image, collection


def fetch_neurovault(max_images=None,
                     collection_terms={}, collection_filter=_empty_filter,
                     image_terms={}, image_filter=_empty_filter,
                     neurovault_dir=None, mode='download_new',
                     download_manager=None, **kwargs):

    if neurovault_dir is None:
        neurovault_dir = _get_dataset_dir('neurovault')

    image_terms = dict(image_terms, **kwargs)

    if download_manager is None:
        download_manager = DownloadManager(max_images=max_images,
                                           data_dir=neurovault_dir)

    scroller = _join_local_and_remote(neurovault_dir=neurovault_dir, mode=mode,
                                      collection_terms=collection_terms,
                                      collection_filter=collection_filter,
                                      image_terms=image_terms,
                                      image_filter=image_filter,
                                      download_manager=download_manager,
                                      max_images=max_images)

    images_meta, collections_meta = zip(*scroller)
    images = [im_meta['local_path'] for im_meta in images_meta]
    return {'images': images,
            'images_meta': images_meta,
            'collections_meta': collections_meta}


def _get_neurovault_keys():
    try:
        meta = _get_neurovault_keys.meta_
    except AttributeError as e:
        meta = None
    if meta is None:
        meta = fetch_neurovault(
            max_images=1, download_manager=BaseDownloadManager(max_images=1))

    return [{k: (object if v is None else type(v)) for
             k, v in meta[doc_type][0].items()} for
            doc_type in ('images_meta', 'collections_meta')]


def show_neurovault_image_keys():
    pprint(_get_neurovault_keys()[0])


def show_neurovault_collection_keys():
    pprint(_get_neurovault_keys()[1])
