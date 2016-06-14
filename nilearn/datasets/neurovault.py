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


def _neurosynth_fetch_words_url():
    return 'http://neurosynth.org/api/v2/decode/'


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


def _get_encoding(resp):
    """Get the encoding of an HTTP response."""
    encoding = None
    try:
        encoding = resp.headers.get_content_charset()
    except AttributeError as e:
        pass
    content_type = resp.headers.get('Content-Type')
    encoding = re.search(r'charset=\b(.+)\b', content_type).group(1)
    return encoding

def _get_batch(query, prefix_msg=''):
    """Given a query, get the response and transform json to python dict.

    Parameters
    ----------
    query: str
    The url from which to get data.

    prefix_msg: str, optional (default='')
    Prefix for all log messages.

    Returns
    -------
    batch: dict or None
    None if download failed, if encoding could not be understood,
    if the response was not in json format or if it did not contain
    one of the required keys 'results' and 'count'.
    Python dict representing the response otherwise.

    """
    request = Request(query)
    opener = build_opener()
    _logger.debug('{}getting new batch: {}'.format(
        prefix_msg, query))
    try:
        resp = opener.open(request)
        encoding = _get_encoding(resp)
        content = resp.read()
        resp.close()
    except Exception as e:
        resp.close()
        _logger.exception(
            'could not download batch from {}'.format(query))
        return None
    try:
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
    """download list of metadata from Neurovault

    Parameters
    ----------
    url: str
    the base url (without the filters) from which to get data

    local_filter: callable, optional (default=_empty_filter)
    Used to filter the results based on their metadata:
    must return True is the result is to be kept and False otherwise.
    Is called with the dict containing the metadata as sole argument.

    query_terms: dict or sequence of pairs or None, optional (default=None)
    Key-value pairs to add to the base url in order to form query.
    If None, nothing is added to the url.

    max_results: int or None, optional (default=None)
    Maximum number of results to fetch; if None, all available data
    that matches the query is fetched.

    batch_size: int or None, optional (default=None)
    Neurovault returns the metadata for hits corresponding to a query
    in batches. batch_size is used to choose the (maximum) number of
    elements in a batch. If None, _default_batch_size() is used.

    prefix_msg: str, optional (default='')
    Prefix for all log messages.

    """
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
    """Special value used to filter terms.

    An instance of this class, as it is used by ResultFilter
    objects, will always be equal to any non-zero value of any
    type (by non-zero) we mean for which bool returns True.

    Instances of this class are only meant to be used as values
    in a query_terms parameter passed to a ResultFilter.

    """
    def __eq__(self, other):
        return bool(other)

    def __req__(self, other):
        return self.__eq__(other)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __rneq__(self, other):
        return self.__neq__(other)


class NotEqual(object):
    """Special value used to filter terms.

    An instance of this class is constructed with NotEqual(obj).
    As it is used by ResultFilter objects, it will allways be equal
    to any value for which obj == value is False.

    Instances of this class are only meant to be used as values
    in a query_terms parameter passed to a ResultFilter.

    """
    def __init__(self, negated):
        """Be equal to any value which is not negated."""
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
    """Easily create callable (local) filters for fetch_neurovault.

    Constructed from a mapping of key-value pairs (optional)
    and a callable filter (also optional),
    instances of this class are meant to be used as image_filter or
    collection_filter parameters for fetch_neurovault.

    Such filters can be combined using
    the logical operators |, &,  ^, not,
    with the usual semantics.

    Key-value pairs can be added by treating a ResultFilter as a
    dictionary: after evaluating res_filter[key] = value, only
    metadata such that metadata[key] == value will pass through the
    filter.

    Attributes
    ----------
    query_terms_: dict
    In order to pass through the filter, metadata must verify
    metadata[key] == value for each key, value pair in query_terms.

    callable_filters_: list of callables
    In addition to (key, value pairs), we can use this attribute to specify
    more elaborate requirements. Called with a dict representing metadata for
    an image or collection, each element of this list returns True if the
    metadata should pass through the filter and False otherwise.

    A dict of metadata will only pass through the filter if it satisfies
    all the query_terms AND all the elements of callable_filters_.

    Methods
    -------
    __init__
    Construct an instance from an initial mapping of key, value pairs
    and an initial callable filter.

    __call__
    Called with a dict representing metadata, returns True if it satisfies
    the requirements expressed by the filter and False otherwise.

    __or__, __and__, __xor__, __not__, and the correspondig reflected operators:
    Used to combine ResultFilter objects.
    Example: metadata will pass through filt1 | filt2 if and only if
    it passes through filt1 or through filt2.

    __getitem__, __setitem__, __delitem__:
    These calls are dispatched to query_terms_. Used to evaluate, add or
    remove elements from the 'key must be value' filters expressed
    in query_terms.

    add_filter:
    Add a callable filter to callable_filters_.

    Returns
    -------
    None

    """
    not_null = NotNull()

    def __init__(self, query_terms={}, callable_filter=_empty_filter, **kwargs):
        """Construct a ResultFilter

        Parameters
        ----------
        query_terms: dict, optional (default={})
        a metadata dict will be blocked by the filter if it does not respect
        metadata[key] == value for all key, value pairs in query_terms.

        callable_filter: callable, optional (default=_empty_filter)
        a metadata dict will be blocked by the filter if it does not respect
        callable_filter(metadata) == True.

        As an alternative to the query_terms dictionary parameter,
        key, value pairs can be passed as keyword arguments.

        """
        query_terms = dict(query_terms, **kwargs)
        self.query_terms_ = query_terms
        self.callable_filters_ = [callable_filter]

    def __call__(self, candidate):
        """Return True if candidate satisfies the requirements.

        Parameters
        ----------
        candidate: dict
        A dictionary representing metadata for a file or a collection,
        to be filtered.

        Returns
        -------
        True if candidates passes through the filter and False otherwise.

        """
        for key, value in self.query_terms_.items():
            if not(value == candidate.get(key)):
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

    def __getitem__(self, item):
        """Get item from query_terms_"""
        return self.query_terms_[item]

    def __setitem__(self, item, value):
        """Set item in query_terms_"""
        self.query_terms_[item] = value

    def __delitem__(self, item):
        """Remove item from query_terms"""
        if item in self.query_terms_:
            del self.query_terms_[item]

    def add_filter(callable_filter):
        """Add a function to the callable_filters_.

        After a call add_filter(additional_filt), in addition
        to all the previous requirements, a candidate must also
        verify additional_filt(candidate) in order to pass through
        the filter.

        """
        self.callable_filters_.append(callable_filter)


def _simple_download(url, target_file, temp_dir):
    """Wrapper around _fetch_file which allows specifying target file name."""
    _logger.debug('downloading file: {}'.format(url))
    downloaded = _fetch_file(url, temp_dir, resume=False,
                             overwrite=True, verbose=0)
    shutil.move(downloaded, target_file)
    _logger.debug(
        'download succeeded, downloaded to: {}'.format(target_file))
    return target_file


def _checked_get_dataset_dir(dataset_name, suggested_dir=None):
    """Wrapper for _get_dataset_dir; expands . and ~ and checks write access"""
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=suggested_dir)
    if not os.access(dataset_dir, os.W_OK):
        raise IOError('Permission denied: {}'.format(dataset_dir))
    return dataset_dir


def _get_temp_dir(suggested_dir=None):
    """Get a sandbox dir in which to download files."""
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    if suggested_dir is None or not os.path.isdir(suggested_dir):
        suggested_dir = mkdtemp()
    if not os.access(suggested_dir, os.W_OK):
        raise IOError('Permission denied: {}'.format(suggested_dir))
    return suggested_dir


def _fetch_neurosynth_words(image_id, target_file, temp_dir):
    """Query Neurosynth for words associated with a map.

    Parameters
    ----------
    image_id: int
    The Neurovault id of the statistical map.

    target_file: str
    Path to the file in which the terms will be stored on disk
    (a json file).

    temp_dir: str
    Path to directory used by _simple_download.
    """
    query = urljoin(_neurosynth_fetch_words_url(),
                    '?neurovault={}'.format(image_id))
    _simple_download(query, target_file, temp_dir)


class BaseDownloadManager(object):
    """Base class for all download managers.

    download managers are used as parameters for
    fetch_neurovault; they download the files and store them
    on disk.

    A BaseDownloadManager does not download anything,
    but increments a counter each time self.image is called,
    and raises a StopIteration exception when the specified
    max number of images has been reached.

    Subclasses should override _collection_hook and
    _image_hook in order to perform the actual work.
    They should not override image as it is responsible for
    stopping the stream of metadata when the max numbers of
    images has been reached.

    Attributes
    ----------
    max_images: int
    Number of calls to self.image after which
    a StopIteration exception will be raised.

    nv_data_dir_: str
    Path to the neurovault home directory.

    Methods
    -------
    collection:
    Called each time metadata for a collection is retrieved.

    image:
    Called each time metadata for an image is retreived.

    _collection_hook, _image_hook:
    Callbacks to be overriden by subclasses. They should perform
    the necessary actions in order to save the relevant data on disk,
    and return the metadata (which they may have modified).

    """
    def __init__(self, neurovault_data_dir=None, max_images=100):
        self.nv_data_dir_ = _checked_get_dataset_dir(
            'neurovault', neurovault_data_dir)
        if max_images is not None and max_images < 0:
            max_images = None
        self.max_images_ = max_images
        self.already_downloaded_ = 0

    def collection(self, collection_info):
        return self._collection_hook(collection_info)

    def image(self, image_info):
        """Stop metadata stream if max_images has been reached."""
        if self.already_downloaded_ == self.max_images_:
            raise StopIteration()
        image_info = self._image_hook(image_info)
        self.already_downloaded_ += 1
        return image_info

    def _collection_hook(self, collection_info):
        """Hook for subclasses."""
        return collection_info

    def _image_hook(self, image_info):
        """Hook for subclasses."""
        return image_info


class DownloadManager(BaseDownloadManager):
    """Store maps, metadata, reduced representations and associated words.

    This download manager stores:
    - in nv_data_dir_: for each collection, a subdirectory
    containing metadata for the collection, the brain maps, the
    metadata for the brain maps, and reduced representations (.npy) files
    of these maps.
    subdirectories are named collection_<Neurovault collection id>
    collection metadata files are collection_<NV collection id>_metadata.json
    maps are named image_<Neurovault image id>.nii.gz
    map metadata files are image_<Neurovault image id>_metadata.json
    reduced representations are image_<NV iamge id>_reduced_rep.npy

    - optionally, in ns_data_dir_: for each image, the words that were
    associated to it by Neurosynth and their weights, as a json file.
    These files are named words_for_image<NV image id>.json

    Attributes
    ----------
    max_images_: int
    number of downloaded images after which the metadata stream is
    stopped (StopIteration is raised).

    nv_data_dir_: str
    Path to the directory in which maps and metadata are stored.

    fetch_ns_: bool
    specifies wether the words should be retreived from Neurosynth.

    ns_data_dir_: str, does not exist if fetch_ns_ is False
    Path to directory in which Neurosynth words are stored.

    temp_dir_: str
    Path to sandbox directory in which files are downloaded before
    being moved to their final destination.

    Methods
    -------
    __init__:
    Specify directories and wether to fetch Neurosynth words.

    collection:
    Receive collection metadata.

    _collection_hook:
    Store collection metadata; creating collection directory if necessary.

    image:
    Receive image metadata, stop data stream if max_images is reached.

    _image_hook:
    Download image, reduced representation if available,
    Neurosynth words if required, and store them on disk.

    """
    def __init__(self, neurovault_data_dir=None, temp_dir=None,
                 fetch_neurosynth_words=False, neurosynth_data_dir=None,
                 max_images=100):
        """Construct DownloadManager.

        Parameters
        ----------
        neurovault_data_dir: str or None, optional (default=None)
        Directory in which to store Neurovault images and metadata.
        if None, a reasonable location is found by _get_dataset_dir.

        temp_dir: str or None, optional (default=None)
        Sandbox directory for downloads.
        if None, a temporary directory is created by tempfile.mkdtemp.

        fetch_neurosynth_words: bool, optional (default=False)
        wether to collect words from Neurosynth.

        neursynth_data_dir: str or None, optional (default=None)
        Directory in which to store Neurosynth words.
        if None, a reasonable location is found by _get_dataset_dir.

        max_images: int, optional (default=100)
        Maximum number of images to download.

        Returns
        -------
        None

        """
        super(DownloadManager, self).__init__(
            neurovault_data_dir=neurovault_data_dir, max_images=max_images)
        self.temp_dir_ = _get_temp_dir(temp_dir)
        self.fetch_ns_ = fetch_neurosynth_words
        if self.fetch_ns_:
            self.ns_data_dir_ = _checked_get_dataset_dir(
                'neurosynth', neurosynth_data_dir)

    def _collection_hook(self, collection_info):
        """Create collection subdir and store metadata.

        Parameters
        ----------
        collection_info: dict
        Collection metadata

        Returns
        -------
        collection_info: dict
        Collection metadata, with local_path added to it.

        """
        collection_id = collection_info['id']
        collection_dir = os.path.join(
            self.nv_data_dir_, 'collection_{}'.format(collection_id))
        collection_info['local_path'] = collection_dir
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(collection_dir,
                                          'collection_metadata.json')
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(collection_info, metadata_file)
        return collection_info

    def _image_hook(self, image_info):
        """Download image, reduced rep (maybe), words (maybe), and store them.

        Parameters
        ----------
        image_info: dict
        Image metadata.

        Returns
        -------
        image_info: dict
        Image metadata, with local_path, reduced_representation_local_path
        (if reduced representation available), and neurosynth_words_local_path
        (if self.fetch_ns_) add to it.

        """
        collection_id = image_info['collection_id']
        collection_dir = os.path.join(
            self.nv_data_dir_, 'collection_{}'.format(collection_id))
        image_id = image_info['id']
        image_url = image_info['file']
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(
            collection_dir, 'image_{}_metadata.json'.format(image_id))
        image_file_path = os.path.join(
            collection_dir, 'image_{}.nii.gz'.format(image_id))
        _simple_download(image_url, image_file_path, self.temp_dir_)
        image_info['local_path'] = image_file_path
        reduced_image_url = image_info.get('reduced_representation')
        if reduced_image_url is not None:
            reduced_image_file = os.path.join(
                collection_dir, 'image_{}_reduced_rep.npy'.format(image_id))
            _simple_download(
                reduced_image_url, reduced_image_file, self.temp_dir_)
            image_info['reduced_representation_local_path'] = reduced_image_file
        if self.fetch_ns_:
            ns_words_file = os.path.join(
                self.ns_data_dir_, 'words_for_image_{}.json'.format(image_id))
            _fetch_neurosynth_words(image_id, ns_words_file, self.temp_dir_)
            image_info['neurosynth_words_local_path'] = ns_words_file
        with open(metadata_file_path, 'w') as metadata_file:
            json.dump(image_info, metadata_file)
        # self.already_downloaded_ is incremented only after
        # this routine returns successfully.
        _logger.debug('already downloaded {} image{}'.format(
            self.already_downloaded_ + 1,
            ('s' if self.already_downloaded_ + 1 > 1 else '')))
        return image_info


# TODO: finish docstring.
def _scroll_server_data(collection_query_terms={},
                        collection_local_filter=_empty_filter,
                        image_query_terms={},
                        image_local_filter=_empty_filter,
                        download_manager=None, max_images=None,
                        metadata_batch_size=None):
    """Return a generator iterating over neurovault.org results for a query."""
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
            if isinstance(e, StopIteration):
                raise
            _logger.exception('_scroll_server_data: bad collection: {}'.format(
                collection))
            bad_collection = True

        if not bad_collection:
            n_im_in_collection = 0
            query = urljoin(_neurovault_collections_url(),
                            '{}/images/'.format(collection['id']))
            images = _scroll_server_results(
                query, query_terms=image_query_terms,
                local_filter=image_local_filter,
                prefix_msg='scroll images from collection {}: '.format(
                    collection['id']),
                batch_size=metadata_batch_size)
            for image in images:
                try:
                    image = download_manager.image(image)
                    yield image, collection
                    n_im_in_collection += 1
                except Exception as e:
                    if isinstance(e, StopIteration):
                        raise
                    _logger.exception(
                        '_scroll_server_data: bad image: {}'.format(image))
            _logger.info(
                'on neurovault.org: '
                '{} image{} matched query in collection {}'.format(
                    (n_im_in_collection if n_im_in_collection else 'no'),
                    ('s' if n_im_in_collection > 1 else ''), collection['id']))


def _json_from_file(filename):
    """Load a json file encoded wit UTF-8."""
    with open(filename, 'rb') as dumped:
        loaded = json.loads(dumped.read().decode('utf-8'))
    return loaded


def _json_add_local_dir(filename):
    """Load a json file and add is parent dir to resulting dict."""
    loaded = _json_from_file(filename)
    loaded.setdefault('local_path', os.path.dirname(filename))
    return loaded


def _json_add_local_path(filename):
    """Load a json file and add its path to resulting dict."""
    loaded = _json_from_file(filename)
    loaded.setdefault('local_path', filename)
    return loaded


# TODO: finish docstring
def _scroll_local_data(neurovault_dir,
                       collection_filter=_empty_filter,
                       image_filter=_empty_filter,
                       max_images=None):
    """Get a generator iterating over local neurovault data matching a query."""
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
    """Move filters handled by the server inside url."""
    local_terms, server_terms = _split_terms(terms, available_on_server)
    local_filter = local_filter & ResultFilter(query_terms=local_terms)
    return server_terms, local_filter


def _prepare_local_scroller(neurovault_dir, collection_terms, collection_filter,
                            image_terms, image_filter, max_images):
    """Construct filters for call to _scroll_local_data."""
    collection_local_filter = (collection_filter
                               & ResultFilter(**collection_terms))
    image_local_filter = (image_filter
                          & ResultFilter(**image_terms))
    local_data = _scroll_local_data(
        neurovault_dir, collection_filter=collection_local_filter,
        image_filter=image_local_filter, max_images=max_images)

    return local_data


# TODO: finish docstring
def _prepare_remote_scroller(collection_terms, collection_filter,
                             image_terms, image_filter,
                             collection_ids, image_ids,
                             download_manager, max_images):
    """Construct filters for call to _scroll_server_data."""
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


# TODO: finish docstring
def _join_local_and_remote(neurovault_dir, mode='download_new',
                           collection_terms={}, collection_filter=_empty_filter,
                           image_terms={}, image_filter=_empty_filter,
                           download_manager=None, max_images=None):
    """Iterate over results found on disk, then those found on neurovault.org"""
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


def default_collection_terms():
    """Return a term filter that excludes empty collections."""
    return {'number_of_images': NotNull()}


def default_image_terms():
    """Return a filter that selects valid, thresholded images in mni space"""
    return {'not_mni': False, 'is_valid': True, 'is_thresholded': False}

# TODO: finish docstring
def fetch_neurovault(max_images=None,
                     collection_terms=default_collection_terms(),
                     collection_filter=_empty_filter,
                     image_terms=default_image_terms(),
                     image_filter=_empty_filter,
                     mode='download_new',
                     neurovault_data_dir=None,
                     neurosynth_data_dir=None, fetch_neurosynth_words=False,
                     download_manager=None, **kwargs):
    """Download data from neurovault.org and neurosynth.org."""
    image_terms = dict(image_terms, **kwargs)

    if download_manager is None:
        download_manager = DownloadManager(
            max_images=max_images,
            neurovault_data_dir=neurovault_data_dir,
            fetch_neurosynth_words=fetch_neurosynth_words,
            neurosynth_data_dir=neurosynth_data_dir)

    scroller = _join_local_and_remote(
        neurovault_dir=download_manager.nv_data_dir_,
        mode=mode,
        collection_terms=collection_terms,
        collection_filter=collection_filter,
        image_terms=image_terms,
        image_filter=image_filter,
        download_manager=download_manager,
        max_images=max_images)

    scroller = list(scroller)
    if not scroller:
        return None
    images_meta, collections_meta = zip(*scroller)
    images = [im_meta.get('local_path') for im_meta in images_meta]
    return {'images': images,
            'images_meta': images_meta,
            'collections_meta': collections_meta}


def _get_neurovault_keys():
    """Return keys found in Neurovault collection and image metadata."""
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
    """Display keys found in Neurovault metadata for an image."""
    pprint(_get_neurovault_keys()[0])


def show_neurovault_collection_keys():
    """Display keys found in Neurovault metadata for a collection."""
    pprint(_get_neurovault_keys()[1])
