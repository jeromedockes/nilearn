import os
from collections import OrderedDict
import tempfile
import shutil
import json

import numpy as np
from nose.tools import (assert_true, assert_false, assert_equal)

from nilearn.datasets import neurovault as nv


try:
    _TemporaryDirectory = tempfile.TemporaryDirectory
except AttributeError:
    # only introduced in tempfile in python3, so use a
    # basic handwritten function if python2
    class _TemporaryDirectory(object):
        def __enter__(self):
            self.temp_dir_ = tempfile.mkdtemp()
            return self.temp_dir_

        def __exit__(self, *args):
            shutil.rmtree(self.temp_dir_)


def test_translate_types_to_sql():
    py_types = {'some_int': int, 'some_float': float,
                'some_str': str, 'some_bool': bool, 'some_dict': dict}
    sql_types = nv._translate_types_to_sql(py_types)
    assert_equal(sql_types['some_int'], 'INTEGER')
    assert_equal(sql_types['some_float'], 'REAL')
    assert_equal(sql_types['some_str'], 'TEXT')
    assert_equal(sql_types['some_bool'], 'INTEGER')
    assert_equal(sql_types['some_dict'], '')


def test_append_filters_to_query():
    query = nv._append_filters_to_query(
        nv._NEUROVAULT_COLLECTIONS_URL,
        OrderedDict([('owner', 'me'), ('DOI', 17)]))
    assert_equal(
        query, 'http://neurovault.org/api/collections/owner=me&DOI=17')


def ignore_connection_errors(func):
    def decorate(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except nv.URLError:
            pass

    return decorate


@ignore_connection_errors
def test_get_encoding():
    request = nv.Request('http://www.google.com')
    opener = nv.build_opener()
    try:
        response = opener.open(request)
    except Exception:
        return
    try:
        nv._get_encoding(response)
    finally:
        response.close()


@ignore_connection_errors
def test_get_batch():
    batch = nv._get_batch(nv._NEUROVAULT_COLLECTIONS_URL)
    assert('results' in batch)
    assert('count' in batch)


@ignore_connection_errors
def test_scroll_server_results():
    result = list(nv._scroll_server_results(nv._NEUROVAULT_COLLECTIONS_URL,
                                            max_results=6, batch_size=3))
    assert_equal(len(result), 6)
    result = list(nv._scroll_server_results(nv._NEUROVAULT_COLLECTIONS_URL,
                                            max_results=3,
                                            local_filter=lambda r: False))
    assert_equal(len(result), 0)


def test_NotNull():
    not_null = nv.NotNull()
    assert_true(not_null == 'a')
    assert_false(not_null == '')
    assert_true('a' == not_null)
    assert_false('' == not_null)
    assert_false(not_null != 'a')
    assert_true(not_null != '')
    assert_false('a' != not_null)
    assert_true('' != not_null)


def test_NotEqual():
    not_equal = nv.NotEqual('a')
    assert_true(not_equal == 'b')
    assert_true(not_equal == 1)
    assert_false(not_equal == 'a')
    assert_true('b' == not_equal)
    assert_true(1 == not_equal)
    assert_false('a' == not_equal)
    assert_false(not_equal != 'b')
    assert_false(not_equal != 1)
    assert_true(not_equal != 'a')
    assert_false('b' != not_equal)
    assert_false(1 != not_equal)
    assert_true('a' != not_equal)


def test_IsIn():
    is_in = nv.IsIn({0, 1})
    assert_true(is_in == 0)
    assert_false(is_in == 2)
    assert_true(0 == is_in)
    assert_false(2 == is_in)
    assert_false(is_in != 0)
    assert_true(is_in != 2)
    assert_false(0 != is_in)
    assert_true(2 != is_in)


def test_ResultFilter():
    filter_0 = nv.ResultFilter(query_terms={'a': 0},
                               callable_filter=lambda d: len(d) < 5,
                               b=1)
    assert_equal(filter_0['a'], 0)
    assert_true(filter_0({'a': 0, 'b': 1, 'c': 2}))
    assert_false(filter_0({'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}))
    assert_false(filter_0({'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0({'a': 1, 'b': 1, 'c': 2}))

    filter_1 = nv.ResultFilter(query_terms={'c': 2})
    filter_1['d'] = nv.NotNull()
    assert_true(filter_1({'c': 2, 'd': 1}))
    assert_false(filter_1({'c': 2, 'd': 0}))
    filter_1['d'] = nv.IsIn({0, 1})
    assert_true(filter_1({'c': 2, 'd': 1}))
    assert_false(filter_1({'c': 2, 'd': 2}))
    filter_1['d'] = nv.NotEqual(nv.IsIn({0, 1}))
    assert_false(filter_1({'c': 2, 'd': 1}))
    assert_true(filter_1({'c': 2, 'd': 3}))
    filter_1.add_filter(lambda d: len(d) > 2)
    assert_false(filter_1({'c': 2, 'd': 3}))
    assert_true(filter_1({'c': 2, 'd': 3, 'e': 4}))


def test_ResultFilter_combinations():
    filter_0 = nv.ResultFilter(a=0, b=1)
    filter_1 = nv.ResultFilter(c=2, d=3)

    filter_0_and_1 = filter_0 & filter_1
    assert_true(filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_false(filter_0_and_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))

    filter_0_or_1 = filter_0 | filter_1
    assert_true(filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_true(filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_true(filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': None}))

    filter_0_xor_1 = filter_0 ^ filter_1
    assert_false(filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_true(filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_true(filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': None}))

    not_filter_0 = ~ filter_0
    assert_true(not_filter_0({}))
    assert_false(not_filter_0({'a': 0, 'b': 1}))


# @with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
# for some reason, when using this tst.tmpdir is None.
# TODO: find out why and use tst.setup_tmpdata.
# In the meanwhile, use TemporaryDirectory
@ignore_connection_errors
def test_simple_download():
    with _TemporaryDirectory() as temp_dir:
        downloaded_file = nv._simple_download(
            'http://neurovault.org/media/images/35/Fig3B_zstat1.nii.gz',
            os.path.join(temp_dir, 'image_35.nii.gz'), temp_dir)
        assert_true(os.path.isfile(downloaded_file))


def test_checked_get_dataset_dir():
    with _TemporaryDirectory() as temp_dir:
        dataset_dir = nv._checked_get_dataset_dir('neurovault', temp_dir)
        assert_true(os.path.samefile(
            dataset_dir, os.path.join(temp_dir, 'neurovault')))


def test_neurovault_directory():
    nv_dir = nv.neurovault_directory()
    assert_true(os.path.isdir(nv_dir))


def test_set_neurovault_directory():
    try:
        with _TemporaryDirectory() as temp_dir:
            dataset_dir = nv.set_neurovault_directory(temp_dir, 'neurovault')
            assert_true(os.path.samefile(
                dataset_dir, os.path.join(temp_dir, 'neurovault')))
    finally:
        nv.set_neurovault_directory(None)


def test_get_temp_dir():
    with _TemporaryDirectory() as temp_dir:
        returned_temp_dir = nv._get_temp_dir(temp_dir)
        assert_true(os.path.samefile(
            returned_temp_dir, temp_dir))
    temp_dir = nv._get_temp_dir()
    try:
        assert_true(os.path.isdir(temp_dir))
    finally:
        shutil.rmtree(temp_dir)


@ignore_connection_errors
def test_fetch_neurosynth_words():
    with _TemporaryDirectory() as temp_dir:
        words_file_name = os.path.join(
            temp_dir, 'neurosynth_words_for_image_110.json')
        nv._fetch_neurosynth_words(
            110, words_file_name, temp_dir)
        with open(words_file_name) as words_file:
            words = json.load(words_file)
            assert_true(words)


def test_neurosynth_words_vectorized():
    n_im = 5
    with _TemporaryDirectory() as temp_dir:
        words_files = [
            os.path.join(temp_dir, 'words_for_image_{}.json'.format(i)) for
            i in range(n_im)]
        words = [str(i) for i in range(n_im)]
        for i, file_name in enumerate(words_files):
            word_weights = np.zeros(n_im)
            word_weights[i] = 1
            words_dict = {'data':
                          {'values':
                           {k: v for k, v in zip(words, word_weights)}}}
            with open(file_name, 'w') as words_file:
                json.dump(words_dict, words_file)
        freq, voc = nv.neurosynth_words_vectorized(words_files)
        assert_equal(freq.shape, (n_im, n_im))
        assert((freq.sum(axis=0) == np.ones(n_im)).all())


def test_BaseDownloadManager():
    download_manager = nv.BaseDownloadManager(max_images=5)

    def g():
        download_manager.image(None)
        for i in range(10):
            download_manager.image({})
            yield i

    assert_equal(len(list(g())), 5)


def test_write_read_metadata():
    metadata = {'relative_path': 'collection_1',
                'absolute_path': '/tmp/collection_1'}
    with _TemporaryDirectory() as temp_dir:
        nv._write_metadata(metadata, os.path.join(temp_dir, 'metadata.json'))
        with open(os.path.join(temp_dir, 'metadata.json')) as meta_file:
            written_metadata = json.load(meta_file)
        assert_true('relative_path' in written_metadata)
        assert_false('absolute_path' in written_metadata)
        read_metadata = nv._add_absolute_paths('/tmp/', written_metadata)
        assert_equal(read_metadata['absolute_path'], '/tmp/collection_1')


def test_download_manager_collection(download_manager=None):
    with _TemporaryDirectory() as data_temp_dir:
        if download_manager is None:
            nv.set_neurovault_directory(data_temp_dir)
            download_manager = nv.DownloadManager(data_temp_dir)
        collection = {'id': 1}
        download_manager.start()
        temp_dir = download_manager.temp_dir_
        assert_true(os.path.isdir(temp_dir))
        with download_manager:
            collection = download_manager.collection(collection)
            with open(os.path.join(
                    collection['absolute_path'],
                    'collection_metadata.json')) as collection_meta_file:
                collection = json.load(collection_meta_file)
        nv.set_neurovault_directory(None)
        assert_equal(collection['id'], 1)
        assert_false(os.path.isdir(temp_dir))


# TODO: remove this
if __name__ == '__main__':
    import collections
    import sys
    for name in dir(sys.modules[__name__]):
        obj = getattr(sys.modules[__name__], name)
        if name.startswith('test') and isinstance(obj, collections.Callable):
            print('calling {}'.format(name))
            obj()
    print('completed successfully')
