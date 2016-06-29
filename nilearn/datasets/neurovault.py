import os
import logging
import warnings
from copy import copy, deepcopy
import shutil
import re
import json
from glob import glob
from tempfile import mkdtemp
from pprint import pprint
import sqlite3
from collections import OrderedDict
import atexit

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import DictVectorizer

from .utils import _fetch_file, _get_dataset_dir
from .._utils.compat import _urllib
urljoin, urlencode = _urllib.parse.urljoin, _urllib.parse.urlencode
Request, build_opener = _urllib.request.Request, _urllib.request.build_opener


# TODO: make docstrings conform to numpy and scikit-learn recommandations
# TODO: tests!!

_NEUROVAULT_BASE_URL = 'http://neurovault.org/api/'
_NEUROVAULT_COLLECTIONS_URL = urljoin(_NEUROVAULT_BASE_URL, 'collections/')
_NEUROVAULT_IMAGES_URL = urljoin(_NEUROVAULT_BASE_URL, 'images/')
_NEUROSYNTH_FETCH_WORDS_URL = 'http://neurosynth.org/api/v2/decode/'

_COL_FILTERS_AVAILABLE_ON_SERVER = {'DOI', 'name', 'owner'}
_IM_FILTERS_AVAILABLE_ON_SERVER = set()

_DEFAULT_BATCH_SIZE = 100

_PY_TO_SQL_TYPE = {int: 'INTEGER', float: 'REAL', str: 'TEXT'}

_IMAGE_BASIC_FIELDS = OrderedDict()
_IMAGE_BASIC_FIELDS['id'] = int
_IMAGE_BASIC_FIELDS['name'] = str
_IMAGE_BASIC_FIELDS['relative_path'] = str
_IMAGE_BASIC_FIELDS['absolute_path'] = str
_IMAGE_BASIC_FIELDS['collection_id'] = int
_IMAGE_BASIC_FIELDS['collection'] = str
_IMAGE_BASIC_FIELDS['add_date'] = str
_IMAGE_BASIC_FIELDS['modify_date'] = str
_IMAGE_BASIC_FIELDS['image_type'] = str
_IMAGE_BASIC_FIELDS['map_type'] = str
_IMAGE_BASIC_FIELDS['url'] = str
_IMAGE_BASIC_FIELDS['file'] = str
_IMAGE_BASIC_FIELDS['file_size'] = int
_IMAGE_BASIC_FIELDS['is_thresholded'] = int
_IMAGE_BASIC_FIELDS['is_valid'] = int
_IMAGE_BASIC_FIELDS['modality'] = str
_IMAGE_BASIC_FIELDS['not_mni'] = int
_IMAGE_BASIC_FIELDS['description'] = str
_IMAGE_BASIC_FIELDS['brain_coverage'] = float
_IMAGE_BASIC_FIELDS['perc_bad_voxels'] = float
_IMAGE_BASIC_FIELDS['perc_voxels_outside'] = float
_IMAGE_BASIC_FIELDS['reduced_representation'] = str
_IMAGE_BASIC_FIELDS['reduced_representation_relative_path'] = str
_IMAGE_BASIC_FIELDS['reduced_representation_absolute_path'] = str
_IMAGE_BASIC_FIELDS['neurosynth_words_relative_path'] = str
_IMAGE_BASIC_FIELDS['neurosynth_words_absolute_path'] = str


def _translate_types_to_sql(fields_dict):
    sql_fields = OrderedDict()
    for k, v in fields_dict.items():
        sql_fields[k] = _PY_TO_SQL_TYPE.get(v, '')
    return sql_fields


_IMAGE_BASIC_FIELDS_SQL = _translate_types_to_sql(_IMAGE_BASIC_FIELDS)

_COLLECTION_BASIC_FIELDS = OrderedDict()
_COLLECTION_BASIC_FIELDS['id'] = int
_COLLECTION_BASIC_FIELDS['relative_path'] = str
_COLLECTION_BASIC_FIELDS['absolute_path'] = str
_COLLECTION_BASIC_FIELDS['DOI'] = str
_COLLECTION_BASIC_FIELDS['name'] = str
_COLLECTION_BASIC_FIELDS['add_date'] = str
_COLLECTION_BASIC_FIELDS['modify_date'] = str
_COLLECTION_BASIC_FIELDS['number_of_images'] = int
_COLLECTION_BASIC_FIELDS['url'] = str
_COLLECTION_BASIC_FIELDS['owner'] = int
_COLLECTION_BASIC_FIELDS['owner_name'] = str
_COLLECTION_BASIC_FIELDS['contributors'] = str

_COLLECTION_BASIC_FIELDS_SQL = _translate_types_to_sql(
    _COLLECTION_BASIC_FIELDS)

_ALL_IMAGE_FIELDS = copy(_IMAGE_BASIC_FIELDS)

_ALL_IMAGE_FIELDS['Action Observation'] = str
_ALL_IMAGE_FIELDS['Acupuncture'] = str
_ALL_IMAGE_FIELDS['Age'] = str
_ALL_IMAGE_FIELDS['Anti-Saccades'] = str
_ALL_IMAGE_FIELDS['Braille Reading'] = str
_ALL_IMAGE_FIELDS['Breath-Holding'] = str
_ALL_IMAGE_FIELDS['CIAS'] = str
_ALL_IMAGE_FIELDS['Chewing/Swallowing'] = str
_ALL_IMAGE_FIELDS['Classical Conditioning'] = str
_ALL_IMAGE_FIELDS['Counting/Calculation'] = str
_ALL_IMAGE_FIELDS['Cued Explicit Recognition'] = str
_ALL_IMAGE_FIELDS['Deception Task'] = str
_ALL_IMAGE_FIELDS['Deductive Reasoning'] = str
_ALL_IMAGE_FIELDS['Delay Discounting Task'] = str
_ALL_IMAGE_FIELDS['Delayed Match To Sample'] = str
_ALL_IMAGE_FIELDS['Divided Auditory Attention'] = str
_ALL_IMAGE_FIELDS['Drawing'] = str
_ALL_IMAGE_FIELDS['Eating/Drinking'] = str
_ALL_IMAGE_FIELDS['Encoding'] = str
_ALL_IMAGE_FIELDS['Episodic Recall'] = str
_ALL_IMAGE_FIELDS['Face Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Film Viewing'] = str
_ALL_IMAGE_FIELDS['Finger Tapping'] = str
_ALL_IMAGE_FIELDS['Fixation'] = str
_ALL_IMAGE_FIELDS['Flanker Task'] = str
_ALL_IMAGE_FIELDS['Flashing Checkerboard'] = str
_ALL_IMAGE_FIELDS['Flexion/Extension'] = str
_ALL_IMAGE_FIELDS['Free Word List Recall'] = str
_ALL_IMAGE_FIELDS['Go/No-Go'] = str
_ALL_IMAGE_FIELDS['Grasping'] = str
_ALL_IMAGE_FIELDS['Imagined Movement'] = str
_ALL_IMAGE_FIELDS['Imagined Objects/Scenes'] = str
_ALL_IMAGE_FIELDS['Isometric Force'] = str
_ALL_IMAGE_FIELDS['Mental Rotation'] = str
_ALL_IMAGE_FIELDS['Micturition Task'] = str
_ALL_IMAGE_FIELDS['Music Comprehension/Production'] = str
_ALL_IMAGE_FIELDS['Naming Covert)'] = str
_ALL_IMAGE_FIELDS['Naming Overt)'] = str
_ALL_IMAGE_FIELDS['Non-Painful Electrical Stimulation'] = str
_ALL_IMAGE_FIELDS['Non-Painful Thermal Stimulation'] = str
_ALL_IMAGE_FIELDS['Oddball Discrimination'] = str
_ALL_IMAGE_FIELDS['Olfactory Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Orthographic Discrimination'] = str
_ALL_IMAGE_FIELDS['Pain Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['PainLevel'] = str
_ALL_IMAGE_FIELDS['Paired Associate Recall'] = str
_ALL_IMAGE_FIELDS['Passive Listening'] = str
_ALL_IMAGE_FIELDS['Passive Viewing'] = str
_ALL_IMAGE_FIELDS['Phonological Discrimination'] = str
_ALL_IMAGE_FIELDS['Pitch Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Pointing'] = str
_ALL_IMAGE_FIELDS['Posner Task'] = str
_ALL_IMAGE_FIELDS['Reading Covert)'] = str
_ALL_IMAGE_FIELDS['Reading Overt)'] = str
_ALL_IMAGE_FIELDS['Recitation/Repetition Covert)'] = str
_ALL_IMAGE_FIELDS['Recitation/Repetition Overt)'] = str
_ALL_IMAGE_FIELDS['Rest'] = str
_ALL_IMAGE_FIELDS['Reward Task'] = str
_ALL_IMAGE_FIELDS['Saccades'] = str
_ALL_IMAGE_FIELDS['Semantic Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Sequence Recall/Learning'] = str
_ALL_IMAGE_FIELDS['Sex'] = str
_ALL_IMAGE_FIELDS['Simon Task'] = str
_ALL_IMAGE_FIELDS['Sleep'] = str
_ALL_IMAGE_FIELDS['Spatial/Location Discrimination'] = str
_ALL_IMAGE_FIELDS['Sternberg Task'] = str
_ALL_IMAGE_FIELDS['Stroop Task'] = str
_ALL_IMAGE_FIELDS['SubjectID'] = str
_ALL_IMAGE_FIELDS['Subjective Emotional Picture Discrimination'] = str
_ALL_IMAGE_FIELDS['Syntactic Discrimination'] = str
_ALL_IMAGE_FIELDS['Tactile Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Task Switching'] = str
_ALL_IMAGE_FIELDS['Theory of Mind Task'] = str
_ALL_IMAGE_FIELDS['Tone Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Tower of London'] = str
_ALL_IMAGE_FIELDS['Transcranial Magnetic Stimulation'] = str
_ALL_IMAGE_FIELDS['Vibrotactile Monitor/Discrimination'] = str
_ALL_IMAGE_FIELDS['Video Games'] = str
_ALL_IMAGE_FIELDS['Visual Distractor/Visual Attention'] = str
_ALL_IMAGE_FIELDS['Visual Pursuit/Tracking'] = str
_ALL_IMAGE_FIELDS['Whistling'] = str
_ALL_IMAGE_FIELDS['Wisconsin Card Sorting Test'] = str
_ALL_IMAGE_FIELDS['Word Generation Covert)'] = str
_ALL_IMAGE_FIELDS['Word Generation Overt)'] = str
_ALL_IMAGE_FIELDS['Word Stem Completion Covert)'] = str
_ALL_IMAGE_FIELDS['Word Stem Completion Overt)'] = str
_ALL_IMAGE_FIELDS['Writing'] = str
_ALL_IMAGE_FIELDS['analysis_level'] = str
_ALL_IMAGE_FIELDS['cognitive_contrast_cogatlas'] = str
_ALL_IMAGE_FIELDS['cognitive_contrast_cogatlas_id'] = str
_ALL_IMAGE_FIELDS['cognitive_paradigm_cogatlas'] = str
_ALL_IMAGE_FIELDS['cognitive_paradigm_cogatlas_id'] = str
_ALL_IMAGE_FIELDS['contrast_definition'] = str
_ALL_IMAGE_FIELDS['contrast_definition_cogatlas'] = str
_ALL_IMAGE_FIELDS['data'] = dict
_ALL_IMAGE_FIELDS['figure'] = str
_ALL_IMAGE_FIELDS['label_description_file'] = str
_ALL_IMAGE_FIELDS['n-back'] = str
_ALL_IMAGE_FIELDS['nidm_results'] = str
_ALL_IMAGE_FIELDS['nidm_results_ttl'] = str
_ALL_IMAGE_FIELDS['number_of_subjects'] = int
_ALL_IMAGE_FIELDS['smoothness_fwhm'] = float
_ALL_IMAGE_FIELDS['statistic_parameters'] = float
_ALL_IMAGE_FIELDS['thumbnail'] = str
_ALL_IMAGE_FIELDS['type'] = str

_ALL_IMAGE_FIELDS_SQL = _translate_types_to_sql(_ALL_IMAGE_FIELDS)

_ALL_COLLECTION_FIELDS = copy(_COLLECTION_BASIC_FIELDS)
_ALL_COLLECTION_FIELDS['acquisition_orientation'] = str
_ALL_COLLECTION_FIELDS['authors'] = str
_ALL_COLLECTION_FIELDS['autocorrelation_model'] = str
_ALL_COLLECTION_FIELDS['b0_unwarping_software'] = str
_ALL_COLLECTION_FIELDS['coordinate_space'] = str
_ALL_COLLECTION_FIELDS['description'] = str
_ALL_COLLECTION_FIELDS['doi_add_date'] = str
_ALL_COLLECTION_FIELDS['echo_time'] = float
_ALL_COLLECTION_FIELDS['field_of_view'] = float
_ALL_COLLECTION_FIELDS['field_strength'] = float
_ALL_COLLECTION_FIELDS['flip_angle'] = float
_ALL_COLLECTION_FIELDS['full_dataset_url'] = str
_ALL_COLLECTION_FIELDS['functional_coregistered_to_structural'] = bool
_ALL_COLLECTION_FIELDS['functional_coregistration_method'] = str
_ALL_COLLECTION_FIELDS['group_comparison'] = bool
_ALL_COLLECTION_FIELDS['group_description'] = str
_ALL_COLLECTION_FIELDS['group_estimation_type'] = str
_ALL_COLLECTION_FIELDS['group_inference_type'] = str
_ALL_COLLECTION_FIELDS['group_model_multilevel'] = str
_ALL_COLLECTION_FIELDS['group_model_type'] = str
_ALL_COLLECTION_FIELDS['group_modeling_software'] = str
_ALL_COLLECTION_FIELDS['group_repeated_measures'] = bool
_ALL_COLLECTION_FIELDS['group_repeated_measures_method'] = str
_ALL_COLLECTION_FIELDS['handedness'] = str
_ALL_COLLECTION_FIELDS['hemodynamic_response_function'] = str
_ALL_COLLECTION_FIELDS['high_pass_filter_method'] = str
_ALL_COLLECTION_FIELDS['inclusion_exclusion_criteria'] = str
_ALL_COLLECTION_FIELDS['interpolation_method'] = str
_ALL_COLLECTION_FIELDS['intersubject_registration_software'] = str
_ALL_COLLECTION_FIELDS['intersubject_transformation_type'] = str
_ALL_COLLECTION_FIELDS['intrasubject_estimation_type'] = str
_ALL_COLLECTION_FIELDS['intrasubject_model_type'] = str
_ALL_COLLECTION_FIELDS['intrasubject_modeling_software'] = str
_ALL_COLLECTION_FIELDS['journal_name'] = str
_ALL_COLLECTION_FIELDS['length_of_blocks'] = float
_ALL_COLLECTION_FIELDS['length_of_runs'] = float
_ALL_COLLECTION_FIELDS['length_of_trials'] = str
_ALL_COLLECTION_FIELDS['matrix_size'] = int
_ALL_COLLECTION_FIELDS['motion_correction_interpolation'] = str
_ALL_COLLECTION_FIELDS['motion_correction_metric'] = str
_ALL_COLLECTION_FIELDS['motion_correction_reference'] = str
_ALL_COLLECTION_FIELDS['motion_correction_software'] = str
_ALL_COLLECTION_FIELDS['nonlinear_transform_type'] = str
_ALL_COLLECTION_FIELDS['number_of_experimental_units'] = int
_ALL_COLLECTION_FIELDS['number_of_imaging_runs'] = int
_ALL_COLLECTION_FIELDS['number_of_rejected_subjects'] = int
_ALL_COLLECTION_FIELDS['object_image_type'] = str
_ALL_COLLECTION_FIELDS['optimization'] = bool
_ALL_COLLECTION_FIELDS['optimization_method'] = str
_ALL_COLLECTION_FIELDS['order_of_acquisition'] = str
_ALL_COLLECTION_FIELDS['order_of_preprocessing_operations'] = str
_ALL_COLLECTION_FIELDS['orthogonalization_description'] = str
_ALL_COLLECTION_FIELDS['paper_url'] = str
_ALL_COLLECTION_FIELDS['parallel_imaging'] = str
_ALL_COLLECTION_FIELDS['proportion_male_subjects'] = float
_ALL_COLLECTION_FIELDS['pulse_sequence'] = str
_ALL_COLLECTION_FIELDS['quality_control'] = str
_ALL_COLLECTION_FIELDS['repetition_time'] = float
_ALL_COLLECTION_FIELDS['resampled_voxel_size'] = float
_ALL_COLLECTION_FIELDS['scanner_make'] = str
_ALL_COLLECTION_FIELDS['scanner_model'] = str
_ALL_COLLECTION_FIELDS['skip_distance'] = float
_ALL_COLLECTION_FIELDS['slice_thickness'] = float
_ALL_COLLECTION_FIELDS['slice_timing_correction_software'] = str
_ALL_COLLECTION_FIELDS['smoothing_fwhm'] = float
_ALL_COLLECTION_FIELDS['smoothing_type'] = str
_ALL_COLLECTION_FIELDS['software_package'] = str
_ALL_COLLECTION_FIELDS['software_version'] = str
_ALL_COLLECTION_FIELDS['subject_age_max'] = float
_ALL_COLLECTION_FIELDS['subject_age_mean'] = float
_ALL_COLLECTION_FIELDS['subject_age_min'] = float
_ALL_COLLECTION_FIELDS['target_resolution'] = float
_ALL_COLLECTION_FIELDS['target_template_image'] = str
_ALL_COLLECTION_FIELDS['transform_similarity_metric'] = str
_ALL_COLLECTION_FIELDS['type_of_design'] = str
_ALL_COLLECTION_FIELDS['used_b0_unwarping'] = bool
_ALL_COLLECTION_FIELDS['used_dispersion_derivatives'] = bool
_ALL_COLLECTION_FIELDS['used_high_pass_filter'] = bool
_ALL_COLLECTION_FIELDS['used_intersubject_registration'] = bool
_ALL_COLLECTION_FIELDS['used_motion_correction'] = bool
_ALL_COLLECTION_FIELDS['used_motion_regressors'] = bool
_ALL_COLLECTION_FIELDS['used_motion_susceptibiity_correction'] = bool
_ALL_COLLECTION_FIELDS['used_orthogonalization'] = bool
_ALL_COLLECTION_FIELDS['used_reaction_time_regressor'] = bool
_ALL_COLLECTION_FIELDS['used_slice_timing_correction'] = bool
_ALL_COLLECTION_FIELDS['used_smoothing'] = bool
_ALL_COLLECTION_FIELDS['used_temporal_derivatives'] = bool

_ALL_COLLECTION_FIELDS_SQL = _translate_types_to_sql(_ALL_COLLECTION_FIELDS)


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


def _append_filters_to_query(query, filters):
    """encode dict or sequence of key-value pairs into a URL query string"""
    if not filters:
        return query
    new_query = urljoin(
        query, urlencode(filters))
    return new_query


def _empty_filter(arg):
    return True


def _get_encoding(resp):
    """Get the encoding of an HTTP response."""
    try:
        return resp.headers.get_content_charset()
    except AttributeError as e:
        pass
    content_type = resp.headers.get('Content-Type', '')
    match = re.search(r'charset=\b(.+)\b', content_type)
    if match is None:
        return None
    return match.group(1)


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
    elements in a batch. If None, _DEFAULT_BATCH_SIZE is used.

    prefix_msg: str, optional (default='')
    Prefix for all log messages.

    """
    query = _append_filters_to_query(url, query_terms)
    if batch_size is None:
        batch_size = _DEFAULT_BATCH_SIZE
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


class IsIn(object):
    def __init__(self, accepted):
        self.accepted_ = accepted

    def __eq__(self, other):
        return other in self.accepted_

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

    __or__, __and__, __xor__, __not__,
    and the correspondig reflected operators:
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

    def __init__(self, query_terms={},
                 callable_filter=_empty_filter, **kwargs):
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

    def add_filter(self, callable_filter):
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


def _checked_get_dataset_dir(dataset_name, suggested_dir=None,
                             write_required=False):
    """Wrapper for _get_dataset_dir; expands . and ~ and checks write access"""
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=suggested_dir)
    if not write_required:
        return dataset_dir
    if not os.access(dataset_dir, os.W_OK):
        raise IOError('Permission denied: {}'.format(dataset_dir))
    return dataset_dir


def neurovault_directory(suggested_path=None):
    """Return path to neurovault directory on filesystem."""
    try:
        if neurovault_directory.directory_path_ is not None:
            return neurovault_directory.directory_path_
    except AttributeError:
        pass
    neurovault_directory.directory_path_ = _checked_get_dataset_dir(
        'neurovault', suggested_path)
    assert(neurovault_directory.directory_path_ is not None)
    refresh_db()
    return neurovault_directory.directory_path_


def set_neurovault_directory(new_dir):
    """Set the default neurovault directory to a new location."""
    try:
        del neurovault_directory.directory_path_
    except Exception as e:
        pass
    close_database_connection()
    return neurovault_directory(new_dir)


def neurovault_metadata_db_path(**kwargs):
    return os.path.join(neurovault_directory(**kwargs),
                        '.neurovault_metadata.db')


def _get_temp_dir(suggested_dir=None):
    """Get a sandbox dir in which to download files."""
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    if (suggested_dir is None or
        not os.path.isdir(suggested_dir) or
        not os.access(suggested_dir, os.W_OK)):
        suggested_dir = mkdtemp()
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
    query = urljoin(_NEUROSYNTH_FETCH_WORDS_URL,
                    '?neurovault={}'.format(image_id))
    _simple_download(query, target_file, temp_dir)


def neurosynth_words_vectorized(word_files):
    words = []
    for file_name in word_files:
        try:
            with open(file_name) as word_file:
                info = json.load(word_file)
                words.append(info['data']['values'])
        except Exception as e:
            _logger.warning(
                'could not load words from file {}'.format(file_name))
    vectorizer = DictVectorizer()
    frequencies = vectorizer.fit_transform(words)
    vocabulary = vectorizer.feature_names_
    return frequencies, vocabulary


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
        self.nv_data_dir_ = neurovault_directory(neurovault_data_dir)
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
        if image_info is not None:
            self.already_downloaded_ += 1
        return image_info

    def update_image(self, image_info):
        return image_info

    def update_collection(self, collection_info):
        return collection_info

    def update(self, image_info, collection_info):
        image_info = self.update_image(image_info)
        collection_info = self.update_collection(collection_info)
        return image_info, collection_info

    def _collection_hook(self, collection_info):
        """Hook for subclasses."""
        return collection_info

    def _image_hook(self, image_info):
        """Hook for subclasses."""
        return image_info

    def start(self):
        pass

    def finish(self):
        pass

    def write_ok(self):
        return os.access(self.nv_data_dir_, os.W_OK)


def _write_metadata(metadata, file_name):
    metadata = {k: v for k, v in metadata.items()
                if not re.search(r'absolute', k)}
    with open(file_name, 'w') as metadata_file:
        json.dump(metadata, metadata_file)


def _add_absolute_paths(root_dir, metadata, force=True):
    set_func = metadata.__setitem__ if force else metadata.setdefault
    absolute_paths = {}
    for name, value in metadata.items():
        match = re.match(r'(.*)relative_path(.*)', name)
        if match is not None:
            abs_name = '{}absolute_path{}'.format(*match.groups())
            absolute_paths[abs_name] = os.path.join(root_dir, value)
    for name, value in absolute_paths.items():
        set_func(name, value)
    return metadata


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
                 fetch_neurosynth_words=False, max_images=100):
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
        collection_name = 'collection_{}'.format(collection_id)
        collection_dir = os.path.join(self.nv_data_dir_, collection_name)
        collection_info['relative_path'] = collection_name
        collection_info['absolute_path'] = collection_dir
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(collection_dir,
                                          'collection_metadata.json')
        _write_metadata(collection_info, metadata_file_path)
        return collection_info

    def _add_words(self, image_info):
        if self.fetch_ns_:
            collection_absolute_path = os.path.dirname(
                image_info['absolute_path'])
            collection_relative_path = os.path.basename(
                collection_absolute_path)
            ns_words_file_name = 'neurosynth_words_for_image_{}.json'.format(
                image_info['id'])
            ns_words_relative_path = os.path.join(collection_relative_path,
                                                  ns_words_file_name)
            ns_words_absolute_path = os.path.join(collection_absolute_path,
                                                  ns_words_file_name)
            image_info[
                'neurosynth_words_relative_path'] = ns_words_relative_path
            image_info[
                'neurosynth_words_absolute_path'] = ns_words_absolute_path

            if not os.path.isfile(ns_words_absolute_path):
                _fetch_neurosynth_words(image_info['id'],
                                        ns_words_absolute_path, self.temp_dir_)
        return image_info

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
        collection_relative_path = 'collection_{}'.format(collection_id)
        collection_absolute_path = os.path.join(
            self.nv_data_dir_, collection_relative_path)
        if not os.path.isdir(collection_absolute_path):
            os.makedirs(collection_absolute_path)
        image_id = image_info['id']
        image_url = image_info['file']
        image_file_name = 'image_{}.nii.gz'.format(image_id)
        image_relative_path = os.path.join(
            collection_relative_path, image_file_name)
        image_absolute_path = os.path.join(
            collection_absolute_path, image_file_name)
        _simple_download(image_url, image_absolute_path, self.temp_dir_)
        image_info['absolute_path'] = image_absolute_path
        image_info['relative_path'] = image_relative_path
        reduced_image_url = image_info.get('reduced_representation')
        if reduced_image_url is not None:
            reduced_image_name = 'image_{}_reduced_rep.npy'.format(image_id)
            reduced_image_relative_path = os.path.join(
                collection_relative_path, reduced_image_name)
            reduced_image_absolute_path = os.path.join(
                collection_absolute_path, reduced_image_name)
            _simple_download(
                reduced_image_url, reduced_image_absolute_path, self.temp_dir_)
            image_info['reduced_representation'
                       '_relative_path'] = reduced_image_relative_path
            image_info['reduced_representation'
                       '_absolute_path'] = reduced_image_absolute_path
        image_info = self._add_words(image_info)
        metadata_file_path = os.path.join(
            collection_absolute_path, 'image_{}_metadata.json'.format(
                image_id))
        _write_metadata(image_info, metadata_file_path)
        # self.already_downloaded_ is incremented only after
        # this routine returns successfully.
        _logger.debug('already downloaded {} image{}'.format(
            self.already_downloaded_ + 1,
            ('s' if self.already_downloaded_ + 1 > 1 else '')))
        return image_info

    def update_image(self, image_info):
        image_info = self._add_words(image_info)
        metadata_file_path = os.path.join(
            os.path.dirname(image_info['absolute_path']),
            'image_{}_metadata.json'.format(image_info['id']))
        _write_metadata(image_info, metadata_file_path)
        return image_info


class SQLiteDownloadManager(DownloadManager):

    def __init__(self, image_fields=_IMAGE_BASIC_FIELDS_SQL.keys(),
                 collection_fields=_COLLECTION_BASIC_FIELDS_SQL.keys(),
                 **kwargs):
        super(SQLiteDownloadManager, self).__init__(**kwargs)
        self.db_file_ = neurovault_metadata_db_path()
        self.connection_ = None
        self.cursor_ = None
        self.im_fields_ = _filter_field_names(image_fields,
                                              _ALL_IMAGE_FIELDS_SQL)
        self.col_fields_ = _filter_field_names(collection_fields,
                                               _ALL_COLLECTION_FIELDS_SQL)
        self._update_sql_statements()

    def _update_sql_statements(self):
        self.im_insert_ = _get_insert_string('images', self.im_fields_)
        self.col_insert_ = _get_insert_string('collections', self.col_fields_)
        self.im_update_ = _get_update_string('images', self.im_fields_)
        self.col_update_ = _get_update_string('collections', self.col_fields_)

    def _add_to_collections(self, collection_info):
        values = [collection_info.get(field) for field in self.col_fields_]
        try:
            self.cursor_.execute(self.col_insert_, values)
        except sqlite3.IntegrityError:
            self.cursor_.execute(self.col_update_, values)
        return collection_info

    def _collection_hook(self, collection_info):
        collection_info = super(SQLiteDownloadManager, self)._collection_hook(
            collection_info)
        collection_info = self._add_to_collections(collection_info)
        return collection_info

    def _add_to_images(self, image_info):
        values = [image_info.get(field) for field in self.im_fields_]
        try:
            self.cursor_.execute(self.im_insert_, values)
        except sqlite3.IntegrityError:
            self.cursor_.execute(self.im_update_, values)
        return image_info

    def _image_hook(self, image_info):
        image_info = super(SQLiteDownloadManager, self)._image_hook(
            image_info)
        image_info = self._add_to_images(image_info)
        return image_info

    def update_image(self, image_info):
        super(SQLiteDownloadManager, self).update_image(image_info)
        return self._add_to_images(image_info)

    def update_collection(self, collection_info):
        super(SQLiteDownloadManager, self).update_collection(collection_info)
        return self._add_to_collections(collection_info)

    def start(self):
        self.finish()
        self.connection_ = sqlite3.connect(self.db_file_)
        self.connection_.row_factory = sqlite3.Row
        self.cursor_ = self.connection_.cursor()
        self._update_schema()

    def _update_schema(self):
        if not _nv_schema_exists(self.cursor_):
            self.cursor_ = _create_schema(
                self.cursor_, self.im_fields_, self.col_fields_)
            return

        for table, col_names, ref_names in [
                ('images', self.im_fields_, _ALL_IMAGE_FIELDS_SQL),
                ('collections', self.col_fields_, _ALL_COLLECTION_FIELDS_SQL)]:
            existing_columns = table_info(self.cursor_, table)[1]
            existing_columns = dict([c[:2] for c in existing_columns])
            existing_col_names = existing_columns.keys()
            for col_name in set(col_names).difference(existing_col_names):
                _logger.warning(
                    'adding column "{}" to existing table "{}"'.format(
                        col_name, table))
                col_str = _get_columns_string([col_name], ref_names)
                self.cursor_.execute(
                    'ALTER TABLE {} ADD {}'.format(table, col_str))
            col_names_to_add = set(existing_col_names).difference(col_names)
            if col_names_to_add:
                _logger.info(
                    'also storing in database values for '
                    'previously existing columns: {} in table {}'.format(
                        ', '.join(col_names_to_add), table))
            col_names.update({name: existing_columns[name] for
                              name in col_names_to_add})
        self._update_sql_statements()
        return

    def finish(self):
        if self.connection_ is None:
            return
        try:
            self.connection_.commit()
            self.connection_.close()
        except Exception as e:
            _logger.exception('error closing db connection')
        self.connection_ = None


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
    download_manager.start()

    collections = _scroll_server_results(_NEUROVAULT_COLLECTIONS_URL,
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
                download_manager.finish()
                raise
            _logger.exception('_scroll_server_data: bad collection: {}'.format(
                collection))
            bad_collection = True

        if not bad_collection:
            n_im_in_collection = 0
            query = urljoin(_NEUROVAULT_COLLECTIONS_URL,
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
                        download_manager.finish()
                        raise
                    _logger.exception(
                        '_scroll_server_data: bad image: {}'.format(image))
            _logger.info(
                'on neurovault.org: '
                '{} image{} matched query in collection {}'.format(
                    (n_im_in_collection if n_im_in_collection else 'no'),
                    ('s' if n_im_in_collection > 1 else ''), collection['id']))

    download_manager.finish()


def _json_from_file(file_name):
    """Load a json file encoded with UTF-8."""
    with open(file_name, 'rb') as dumped:
        loaded = json.loads(dumped.read().decode('utf-8'))
    return loaded


def _json_add_collection_dir(file_name, force=True):
    """Load a json file and add is parent dir to resulting dict."""
    loaded = _json_from_file(file_name)
    set_func = loaded.__setitem__ if force else loaded.setdefault
    dir_path = os.path.dirname(file_name)
    set_func('absolute_path', dir_path)
    set_func('relative_path', os.path.basename(dir_path))
    return loaded


def _json_add_im_files_paths(file_name, force=True):
    """Load a json file and add its path to resulting dict."""
    loaded = _json_from_file(file_name)
    set_func = loaded.__setitem__ if force else loaded.setdefault
    dir_path = os.path.dirname(file_name)
    dir_relative_path = os.path.basename(dir_path)
    image_file_name = 'image_{}.nii.gz'.format(loaded['id'])
    reduced_file_name = 'image_{}_reduced_rep.npy'.format(loaded['id'])
    words_file_name = 'neurosynth_words_for_image_{}.json'.format(loaded['id'])
    set_func('relative_path', os.path.join(dir_relative_path, image_file_name))
    if os.path.isfile(os.path.join(dir_path, reduced_file_name)):
        set_func('reduced_representation_relative_path',
                 os.path.join(dir_relative_path, reduced_file_name))
    if os.path.isfile(os.path.join(dir_path, words_file_name)):
        set_func('neurosynth_words_relative_path',
                 os.path.join(dir_relative_path, words_file_name))
    loaded = _add_absolute_paths(
        os.path.dirname(dir_path), loaded, force=force)
    return loaded


# TODO: finish docstring
def _scroll_local_data(neurovault_dir,
                       collection_filter=_empty_filter,
                       image_filter=_empty_filter,
                       max_images=None):
    """Get an iterator over local neurovault data matching a query."""
    if max_images is not None and max_images < 0:
        max_images = None
    found_images = 0
    neurovault_dir = os.path.abspath(os.path.expanduser(neurovault_dir))
    collections = glob(
        os.path.join(neurovault_dir, '*', 'collection_metadata.json'))

    for collection in filter(collection_filter,
                             map(_json_add_collection_dir, collections)):
        images = glob(os.path.join(
            collection['absolute_path'], 'image_*_metadata.json'))
        for image in filter(image_filter,
                            map(_json_add_im_files_paths, images)):
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


def _prepare_local_scroller(neurovault_dir, collection_terms,
                            collection_filter, image_terms,
                            image_filter, max_images):
    """Construct filters for call to _scroll_local_data."""
    collection_local_filter = (collection_filter &
                               ResultFilter(**collection_terms))
    image_local_filter = (image_filter &
                          ResultFilter(**image_terms))
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
    server_data = _scroll_server_data(
        collection_query_terms=collection_terms,
        collection_local_filter=collection_filter,
        image_query_terms=image_terms,
        image_local_filter=image_filter,
        download_manager=download_manager,
        max_images=max_images)
    return server_data


def _return_same(*args):
    return args


# TODO: finish docstring
def _join_local_and_remote(neurovault_dir, mode='download_new',
                           collection_terms={},
                           collection_filter=_empty_filter,
                           image_terms={}, image_filter=_empty_filter,
                           download_manager=None, max_images=None):
    """Iterate over results from disk, then those found on neurovault.org"""
    if mode not in ['overwrite', 'download_new', 'offline']:
        raise ValueError(
            'supported modes are overwrite,'
            ' download_new, offline; got {}'.format(mode))

    if mode == 'overwrite':
        local_data = tuple()
    else:
        local_data = _prepare_local_scroller(
            neurovault_dir, collection_terms, collection_filter,
            image_terms, image_filter, max_images)
    image_ids, collection_ids = set(), set()

    if download_manager is not None:
        download_manager.start()
        update = download_manager.update
    else:
        update = _return_same

    for image, collection in local_data:
        image, collection = update(image, collection)
        image_ids.add(image['id'])
        collection_ids.add(collection['id'])
        yield image, collection

    if download_manager is not None:
        download_manager.finish()

    if mode == 'offline':
        return
    if max_images is not None and len(image_ids) >= max_images:
        return

    server_data = _prepare_remote_scroller(collection_terms, collection_filter,
                                           image_terms, image_filter,
                                           collection_ids, image_ids,
                                           download_manager, max_images)
    for image, collection in server_data:
        yield image, collection


def basic_collection_terms():
    """Return a term filter that excludes empty collections."""
    return {'number_of_images': NotNull()}


def basic_image_terms():
    """Return a filter that selects valid, thresholded images in mni space"""
    return {'not_mni': False, 'is_valid': True, 'is_thresholded': False}


# TODO: finish docstring
def fetch_neurovault(max_images=None,
                     collection_terms=basic_collection_terms(),
                     collection_filter=_empty_filter,
                     image_terms=basic_image_terms(),
                     image_filter=_empty_filter,
                     mode='download_new',
                     neurovault_data_dir=None,
                     fetch_neurosynth_words=False,
                     download_manager=None, **kwargs):
    """Download data from neurovault.org and neurosynth.org."""
    image_terms = dict(image_terms, **kwargs)

    neurovault_data_dir = neurovault_directory(neurovault_data_dir)
    if mode != 'offline' and not os.access(neurovault_data_dir, os.W_OK):
        warnings.warn("You don't have write access to neurovault dir: {};"
                      "fetch_neurovault is working offline.".format(
                          neurovault_data_dir))
        mode = 'offline'

    if download_manager is None and mode != 'offline':
        download_manager = SQLiteDownloadManager(
            max_images=max_images,
            neurovault_data_dir=neurovault_data_dir,
            fetch_neurosynth_words=fetch_neurosynth_words)

    scroller = _join_local_and_remote(
        neurovault_dir=neurovault_data_dir,
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
    images = [im_meta.get('absolute_path') for im_meta in images_meta]
    result = Bunch(images=images,
                   images_meta=images_meta,
                   collections_meta=collections_meta)
    if fetch_neurosynth_words:
        (result['word_frequencies'],
         result['vocabulary']) = neurosynth_words_vectorized(
             [meta['neurosynth_words_absolute_path'] for meta in images_meta])
    return result


def refresh_db(*args, **kwargs):
    download_manager = SQLiteDownloadManager(*args, **kwargs)
    fetch_neurovault(image_terms={}, collection_terms={},
                     download_manager=download_manager,
                     mode='offline', fetch_neurosynth_words=True)


def _update_metadata_info(collected_info, new_instance):
    """Update a dict of {field: type, #times filled} with new metadata."""
    for k, v in new_instance.items():
        prev_type, prev_nb = collected_info.get(k, (None, 0))
        new_nb = prev_nb + (v is not None)
        new_type = prev_type if v is None else type(v)
        collected_info[k] = new_type, new_nb
    return collected_info


def _get_all_neurovault_keys(max_images=None):
    """Get info about the metadata fields in Neurovault

    Parameters
    ----------
    max_images: int, optional (default=None)
    stop after seeing metadata for max_images images.
    If None, read metadata for all images and collections.

    Returns
    -------
    meta: tuple(dict, dict)
    The first element contains info about image metadata fields,
    the second element about collection metadata fields.
    The image metadata (resp. collection metadata) dict contains '
    ' pairs of the form:
    field_name: (type, number of images (resp. collections) '
    'for which this field is filled)

    """
    try:
        meta = _get_all_neurovault_keys.meta_
    except AttributeError as e:
        meta = None
    if meta is None:
        im_keys = {}
        coll_keys = {}
        seen_colls = set()
        for im, coll in _join_local_and_remote(
                neurovault_dir=neurovault_directory(), max_images=max_images):
            _update_metadata_info(im_keys, im)
            if coll['id'] not in seen_colls:
                seen_colls.add(coll['id'])
                _update_metadata_info(coll_keys, coll)
        meta = im_keys, coll_keys
        _get_all_neurovault_keys.meta_ = meta
    return meta


def show_neurovault_image_keys(max_images=300):
    """Display keys found in Neurovault metadata for images."""
    pprint(_get_all_neurovault_keys(max_images)[0])


def show_neurovault_collection_keys(max_images=300):
    """Display keys found in Neurovault metadata for collections."""
    pprint(_get_all_neurovault_keys(max_images)[1])


def _which_keys_are_unused(max_images=None):
    im_keys, coll_keys = _get_all_neurovault_keys(max_images)
    im_unused, coll_unused = set(), set()
    for k, v in im_keys.items():
        if v[0] is None:
            im_unused.add(k)
    for k, v in coll_keys.items():
        if v[0] is None:
            coll_unused.add(k)
    return im_unused, coll_unused


def _fields_occurences_bar(keys, ax=None, txt_rotation='vertical',
                           fontsize='x-large', **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    width = .8
    name_freq = [(name, info[1]) for (name, info) in keys.items()]
    name, freq = zip(*name_freq)
    name = np.asarray(name)
    freq = np.asarray(freq)
    order = np.argsort(freq)
    ax.bar(range(len(name)), freq[order][::-1], width)
    ax.set_xticks(np.arange(len(name)) + width / 2)
    ax.set_xticklabels(name[order][::-1], rotation=txt_rotation,
                       fontsize=fontsize, **kwargs)


def _prepare_subplots_fields_occurrences():
    gs_im = GridSpec(1, 1, bottom=.65, top=.95)
    gs_col = GridSpec(1, 1, bottom=.2, top=.5)
    ax_im = plt.subplot(gs_im[:])
    ax_im.set_title('image fields')
    ax_col = plt.subplot(gs_col[:])
    ax_col.set_title('column fields', fontsize='xx-large')
    return ax_im, ax_col


def plot_fields_occurrences(max_images=300, **kwargs):
    all_keys = _get_all_neurovault_keys(max_images)
    axis_arr = _prepare_subplots_fields_occurrences()
    for table, ax in zip(all_keys, axis_arr):
        _fields_occurences_bar(table, ax=ax, **kwargs)


def _filter_field_names(required_fields, ref_fields):
    filtered = OrderedDict()
    for field_name in required_fields:
        if field_name in ref_fields:
            filtered[field_name] = ref_fields[field_name]
        else:
            _logger.warning(
                'rejecting unknown column name: {}'.format(field_name))
    return filtered


def _get_columns_string(required_fields, ref_fields):
    fields = ['{} {}'.format(n, v) for
              n, v in _filter_field_names(required_fields, ref_fields).items()]
    return ', '.join(fields)


def _get_insert_string(table_name, fields):
    return "INSERT INTO {} ({}) VALUES ({})".format(
        table_name,
        ', '.join(fields),
        ('?, ' * len(fields))[:-2])


def _get_update_string(table_name, fields):
    set_str = ','.join(["{}=:{}".format(field, field) for field in fields])
    return "UPDATE {} SET {} WHERE id=:id".format(table_name, set_str)


def _table_exists(cursor, table_name):
    cursor.execute("SELECT * FROM sqlite_master WHERE name=?", (table_name,))
    return bool(cursor.fetchall())


def local_database_connection():
    try:
        if local_database_connection.connection_ is not None:
            return local_database_connection.connection_
    except AttributeError:
        pass
    db_path = neurovault_metadata_db_path()
    local_database_connection.connection_ = sqlite3.connect(db_path)
    local_database_connection.connection_.row_factory = sqlite3.Row
    return local_database_connection.connection_


def local_database_cursor():
    return local_database_connection().cursor()


@atexit.register
def close_database_connection():
    try:
        local_database_connection.connection_.commit()
        local_database_connection.connection_.close()
        _logger.info(
            'committed changes to local database and closed connection')
    except (AttributeError, sqlite3.ProgrammingError):
        pass
    except Exception as e:
        _logger.exception()
    local_database_connection.connection_ = None


def _create_schema(cursor, im_fields=_IMAGE_BASIC_FIELDS,
                   col_fields=_COLLECTION_BASIC_FIELDS):
    im_fields = copy(im_fields)
    col_fields = copy(col_fields)
    im_fields.pop('id', None)
    im_fields.pop('collection_id', None)
    col_fields.pop('id', None)
    im_columns = _get_columns_string(im_fields, _ALL_IMAGE_FIELDS_SQL)
    if(im_columns):
        im_columns = ', ' + im_columns
    col_columns = _get_columns_string(col_fields, _ALL_COLLECTION_FIELDS_SQL)
    if(col_columns):
        col_columns = ', ' + col_columns
    im_command = ('CREATE TABLE images '
                  '(id INTEGER PRIMARY KEY, collection_id INTEGER'
                  '{}, FOREIGN KEY(collection_id) '
                  'REFERENCES collections(id))'.format(im_columns))
    col_command = ('CREATE TABLE collections '
                   '(id INTEGER PRIMARY KEY{})'.format(col_columns))
    cursor = cursor.execute(col_command)
    cursor = cursor.execute(im_command)
    cursor.connection.commit()
    return cursor


def _nv_schema_exists(cursor):
    return (_table_exists(cursor, 'images') and
            _table_exists(cursor, 'collections'))


def table_info(cursor, table_name):
    cursor.execute("SELECT sql FROM sqlite_master "
                   "WHERE tbl_name=? AND type='table'", (table_name,))
    table_statement = cursor.fetchone()[0]
    m = re.match(r'CREATE TABLE {} ?\((.*)\)'.format(table_name),
                 table_statement, re.IGNORECASE)
    if not m:
        _logger.error('table_info: could not find column names '
                      'for table {}'.format(table_name))
        return None
    info = m.group(1)
    columns = re.match(r'(.*?)(, FOREIGN.*)?$', info).group(1)
    return info, [pair.split() for pair in columns.split(',')]


def column_names(cursor, table_name):
    columns = table_info(cursor, table_name)[1]
    if columns is None:
        return None
    return next(zip(*columns))


def read_sql_query(query, as_columns=True, curs=None):
    if curs is None:
        curs = local_database_cursor()
    curs.execute(query)
    resp = curs.fetchall()
    if not resp:
        return None
    if not as_columns:
        return resp
    col_names = resp[0].keys()
    cols = zip(*resp)
    cols = map(np.asarray, cols)
    return OrderedDict(zip(col_names, cols))
