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
import errno

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import DictVectorizer

from .utils import _fetch_file, _get_dataset_dir
from .._utils.compat import _urllib
urljoin, urlencode = _urllib.parse.urljoin, _urllib.parse.urlencode
URLError = _urllib.error.URLError
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

_PY_TO_SQL_TYPE = {int: 'INTEGER', bool: 'INTEGER', float: 'REAL', str: 'TEXT'}

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
    """Translate values of a mapping from python to SQL datatypes.

    Given a dictionary which describes metadata fields by mapping
    field names to python types, translate the values (the types) into
    (string representations of) SQL datatypes.

    Parameters
    ----------
    fields_dict : dict
        Maps names of metadata fields to the type of value
        they should contain.

    Returns
    -------
    collections.OrderedDict
        Maps names of metadata fields to the type of value
        they should contain in an SQL table.

    """
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

_ALL_IMAGE_FIELDS['comment'] = str
_ALL_IMAGE_FIELDS['is_bad'] = bool
_ALL_IMAGE_FIELDS['clean_img_relative_path'] = str
_ALL_IMAGE_FIELDS['clean_img_absolute_path'] = str
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

_KNOWN_BAD_COLLECTION_IDS = {16}
_KNOWN_BAD_IMAGE_IDS = {
    96, 97, 98,                    # The following maps are not brain maps
    338, 339,                      # And the following are crap
    335,                           # 335 is a duplicate of 336
    3360, 3362, 3364,              # These are mean images, and not Z maps
    1202, 1163, 1931, 1101, 1099}  # Ugly / obviously not Z maps


class MaxImagesReached(StopIteration):
    """Exception class to signify enough images have been fetched."""
    pass


def prepare_logging(level=logging.DEBUG):
    """Get the root logger and add a handler to it if it doesn't have any.

    Parameters
    ----------
    level : int, optional (default=logging.DEBUG)
        Level of the handler that is added if none exist.
        this handler streams output to the console.

    Returns
    -------
    logging.RootLogger
        The root logger.

    """
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    console_logger = logging.StreamHandler()
    console_logger.setLevel(level)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_logger.setFormatter(formatter)
    logger.addHandler(console_logger)
    return logger


_logger = prepare_logging()


def _append_filters_to_query(query, filters):
    """Encode dict or sequence of key-value pairs into a URL query string

    Parameters
    ----------
    query : str
        URL to which the filters should be appended

    filters : dict or sequence of pairs
        Filters to append to the URL.

    Returns
    -------
    str
        The query with filters appended to it.

    See Also
    --------
    urllib.parse.urlencode

    """
    if not filters:
        return query
    new_query = urljoin(
        query, urlencode(filters))
    return new_query


def _empty_filter(arg):
    """Place holder for a filter which always returns True."""
    return True


def _get_encoding(resp):
    """Get the encoding of an HTTP response.

    Parameters
    ----------
    resp : http.client.HTTPResponse
        Response whose encoding we want to find out.

    Returns
    -------
    str
        str representing the encoding, e.g. 'utf-8'.

    Raises
    ------
    ValueError
        If the response does not specify an encoding.

    """
    try:
        return resp.headers.get_content_charset()
    except AttributeError as e:
        pass
    content_type = resp.headers.get('Content-Type', '')
    match = re.search(r'charset=\b(.+)\b', content_type)
    if match is None:
        raise ValueError(
            'HTTP response encoding not found; headers: {}'.format(
                resp.headers))
    return match.group(1)


def _get_batch(query, prefix_msg=''):
    """Given an URL, get the HTTP response and transform it to python dict.

    The URL is used to send an HTTP GET request and the response is
    transformed into a dict.

    Parameters
    ----------
    query : str
        The URL from which to get data.

    prefix_msg : str, optional (default='')
        Prefix for all log messages.

    Returns
    -------
    batch : dict
        Python dict representing the response's content.

    Raises
    ------
    urllib.error.URLError
        If there was a problem opening the URL.

    ValueError
        If the response could not be decoded, or did not contain
        'results' or 'count'.

    Notes
    -----
    urllib.error.HTTPError is a subclass of URLError.

    """
    request = Request(query)
    opener = build_opener()
    _logger.debug('{}getting new batch: {}'.format(
        prefix_msg, query))
    try:
        resp = opener.open(request)
    except URLError as e:
        _logger.exception(
            'could not download batch from {}'.format(query))
        raise
    try:
        encoding = _get_encoding(resp)
        content = resp.read()
        batch = json.loads(content.decode(encoding))
    except(URLError, ValueError) as e:
        _logger.exception('could not decypher batch from {}'.format(query))
        raise
    finally:
        resp.close()
    for key in ['results', 'count']:
        if batch.get(key) is None:
            msg = ('could not find required key "{}" '
                   'in batch retrieved from {}'.format(key, query))
            _logger.error(msg)
            raise ValueError(msg)

    return batch


def _scroll_server_results(url, local_filter=_empty_filter,
                           query_terms=None, max_results=None,
                           batch_size=None, prefix_msg=''):
    """Download list of metadata from Neurovault.

    Parameters
    ----------
    url : str
        The base url (without the filters) from which to get data.

    local_filter : callable, optional (default=_empty_filter)
        Used to filter the results based on their metadata:
        must return True is the result is to be kept and False otherwise.
        Is called with the dict containing the metadata as sole argument.

    query_terms : dict, sequence of pairs or None, optional (default=None)
        Key-value pairs to add to the base url in order to form query.
        If ``None``, nothing is added to the url.

    max_results: int or None, optional (default=None)
        Maximum number of results to fetch; if ``None``, all available data
        that matches the query is fetched.

    batch_size: int or None, optional (default=None)
        Neurovault returns the metadata for hits corresponding to a query
        in batches. batch_size is used to choose the (maximum) number of
        elements in a batch. If None, ``_DEFAULT_BATCH_SIZE`` is used.

    prefix_msg: str, optional (default='')
        Prefix for all log messages.

    Yields
    ------
    result : dict
        A result in the retrieved batch.

    Raises
    ------
    URLError, ValueError
        If a batch failed to be retrieved.

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


class _SpecialValue(object):
    """Base class for special values used to filter terms.

    Derived classes should override ``__eq__`` in order to create
    objects that can be used for comparisons to particular sets of
    values in filters.

    """
    def __eq__(self, other):
        raise NotImplementedError('Use a derived class for _SpecialValue')

    def __req__(self, other):
        return self.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __rne__(self, other):
        return self.__ne__(other)


class IsNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any null value of any type (by null we mean for which bool
    returns False).

    """
    def __eq__(self, other):
        return not bool(other)


class NotNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any non-zero value of any type (by non-zero we mean for which bool
    returns True).

    """
    def __eq__(self, other):
        return bool(other)


class NotEqual(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with `NotEqual(obj)`. It
    will allways be equal to, and only to, any value for which ``obj
    == value`` is ``False``.

    Parameters
    ----------
    negated : object
        The object from which a candidate should be different in order
        to pass through the filter.

    """
    def __init__(self, negated):
        self.negated_ = negated

    def __eq__(self, other):
        return not self.negated_ == other


class IsIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `IsIn(container)`. It will allways be equal to, and only to, any
    value for which ``value in container`` is ``True``.

    Parameters
    ----------
    accepted : container
        By container we mean any type which exposes a __contains__
        method. A value will pass through the filter if it is present
        in `accepted`.

    """
    def __init__(self, accepted):
        self.accepted_ = accepted

    def __eq__(self, other):
        return other in self.accepted_


class NotIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `NotIn(container)`. It will allways be equal to, and only to, any
    value for which ``value in container`` is ``False``.

    Parameters
    ----------
    rejected : container
        By container we mean any type which exposes a __contains__
        method. A value will pass through the filter if it is absent
        from `rejected`.

    """
    def __init__(self, rejected):
        self.rejected_ = rejected

    def __eq__(self, other):
        return other not in self.rejected_


class ResultFilter(object):

    """Easily create callable (local) filters for ``fetch_neurovault``.

    Constructed from a mapping of key-value pairs (optional) and a
    callable filter (also optional), instances of this class are meant
    to be used as ``image_filter`` or ``collection_filter`` parameters
    for ``fetch_neurovault``.

    Such filters can be combined using their methods ``AND``, ``OR``,
    ``XOR``, and ``NOT``, with the usual semantics.

    Key-value pairs can be added by treating a ``ResultFilter`` as a
    dictionary: after evaluating ``res_filter[key] = value``, only
    metadata such that ``metadata[key] == value`` can pass through the
    filter.

    Parameters
    ----------

    query_terms : dict, optional (default={})
        a metadata dict will be blocked by the filter if it does not
        respect ``metadata[key] == value`` for all ``key``, ``value``
        pairs in `query_terms`.

    callable_filter : callable, optional (default=_empty_filter)
        a ``metadata`` dictionary will be blocked by the filter if
        `callable_filter` does not return ``True`` for ``metadata``.

    As an alternative to the `query_terms` dictionary parameter,
    key, value pairs can be passed as keyword arguments.

    Attributes
    ----------
    query_terms_ : dict
        In order to pass through the filter, metadata must verify
        ``metadata[key] == value`` for each ``key``, ``value`` pair in
        `query_terms_`.

    callable_filters_ : list of callables
        In addition to ``(key, value)`` pairs, we can use this
        attribute to specify more elaborate requirements. Called with
        a dict representing metadata for an image or collection, each
        element of this list returns ``True`` if the metadata should
        pass through the filter and ``False`` otherwise.

    A dict of metadata will only pass through the filter if it
    satisfies all the `query_terms` AND all the elements of
    `callable_filters_`.

    Examples
    --------
    >>> filt = ResultFilter(a=0).AND(ResultFilter(b=1).OR(ResultFilter(b=2)))
    >>> filt({'a': 0, 'b': 1})
    >>> filt({'a': 0, 'b': 0})

    """

    def __init__(self, query_terms={},
                 callable_filter=_empty_filter, **kwargs):
        query_terms = dict(query_terms, **kwargs)
        self.query_terms_ = query_terms
        self.callable_filters_ = [callable_filter]

    def __call__(self, candidate):
        """Return True if candidate satisfies the requirements.

        Parameters
        ----------
        candidate : dict
            A dictionary representing metadata for a file or a
            collection, to be filtered.

        Returns
        -------
        bool
            ``True`` if `candidate` passes through the filter and ``False``
            otherwise.

        """
        for key, value in self.query_terms_.items():
            if not (value == candidate.get(key)):
                return False
        for callable_filter in self.callable_filters_:
            if not callable_filter(candidate):
                return False
        return True

    def OR(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) or filt2(r))
        return new_filter

    def AND(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) and filt2(r))
        return new_filter

    def XOR(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) != filt2(r))
        return new_filter

    def NOT(self):
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

        After a call add_filter(additional_filt), in addition to all
        the previous requirements, a candidate must also verify
        additional_filt(candidate) in order to pass through the
        filter.

        """
        self.callable_filters_.append(callable_filter)


def _simple_download(url, target_file, temp_dir):
    """Wrapper around ``utils._fetch_file``.

    This allows specifying the target file name.

    Parameters
    ----------
    url : str
        URL of the file to download.

    target_file : str
        Location of the downloaded file on filesystem.

    temp_dir : str
        Location of sandbox directory used by ``_fetch_file``.

    Returns
    -------
    target_file : str
        The location in which the file was downloaded.

    Raises
    ------
    URLError, ValueError
        If an error occurred when downloading the file.

    See Also
    --------
    _utils._fetch_file


    Notes
    -----
    It can happen that an HTTP error that occurs inside
    ``_fetch_file`` gets transformed into an ``AttributeError`` when
    we try to set the ``reason`` attribute of the exception raised;
    here we replace it with an ``URLError``.

    """
    _logger.debug('downloading file: {}'.format(url))
    try:
        downloaded = _fetch_file(url, temp_dir, resume=False,
                                 overwrite=True, verbose=0)
    except Exception as e:
        _logger.error('problem downloading file from {}'.format(url))

        # reason is a property of urlib.error.HTTPError objects,
        # but these objects don't have a setter for it, so
        # an HTTPError raised in _fetch_file might be transformed
        # into an AttributeError when we try to set its reason attribute
        if (isinstance(e, AttributeError) and
            e.args[0] == "can't set attribute"):
            raise URLError(
                'HTTPError raised in nilearn.datasets._fetch_file; '
                'then AttributeError when trying to set reason attribute.')
        raise
    shutil.move(downloaded, target_file)
    _logger.debug(
        'download succeeded, downloaded to: {}'.format(target_file))
    return target_file


def _checked_get_dataset_dir(dataset_name, suggested_dir=None,
                             write_required=False):
    """Wrapper for ``_get_dataset_dir``.

    Expands . and ~ and checks write access.

    Parameters
    ----------
    dataset_name : str
        Passed to ``_get_dataset_dir``. Example: ``neurovault``.

    suggested_dir : str
        Desired location of root data directory for all datasets,
        e.g. ``~/home/nilearn_data``.

    write_required : bool, optional (default=False)
        If ``True``, check that the user has write access to the
        chosen data directory and raise ``IOError`` if not.  If
        ``False``, don't check for write permission.

    Returns
    -------
    dataset_dir : str
        The location of the dataset directory in the filesystem.

    Raises
    ------
    IOError
        If `write_required` is set and the user doesn't have write
        access to `dataset_dir`.

    See Also
    --------
    _utils._get_dataset_dir

    """
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=suggested_dir)
    if not write_required:
        return dataset_dir
    if not os.access(dataset_dir, os.W_OK):
        raise IOError('Permission denied: {}'.format(dataset_dir))
    return dataset_dir


def neurovault_directory(suggested_dir=None):
    """Return path to neurovault directory on filesystem.

    A connection to a local database in this directory is open and its
    contents are updated.

    See Also
    --------
    set_neurovault_directory
    refresh_db

    """
    if getattr(neurovault_directory, 'directory_path_', None) is not None:
        return neurovault_directory.directory_path_

    _logger.debug('looking for neurovault directory')
    close_database_connection()
    if suggested_dir is None:
        root_data_dir, dataset_name = None, 'neurovault'
    else:
        suggested_path = suggested_dir.split(os.path.sep)
        dataset_name = suggested_path[-1]
        root_data_dir = os.path.sep.join(suggested_path[:-1])
    neurovault_directory.directory_path_ = _checked_get_dataset_dir(
        dataset_name, root_data_dir)
    assert(neurovault_directory.directory_path_ is not None)
    refresh_db()
    return neurovault_directory.directory_path_


def set_neurovault_directory(new_neurovault_dir=None):
    """Set the default neurovault directory to a new location.

    If the preferred directory is changed, if a connection to a local
    database was open, it is closed; a connection is open to a
    database in the new directory and its contents are updated.

    Parameters
    ----------
    new_neurovault_dir : str, optional (default=None)
        Suggested path for neurovault directory.
        The default value ``None`` means reset neurovault directory
        path to its default value.

    Returns
    -------

    neurovault_directory.directory_path_ : str
        The new neurovault directory used by default by all functions.

    See Also
    --------
    neurovault_directory
    refresh_db
    _checked_get_dataset_dir

    """
    neurovault_directory.directory_path_ = None
    return neurovault_directory(new_neurovault_dir)


def neurovault_metadata_db_path():
    """Get location of sqlite file holding Neurovault metadata."""
    db_path = os.path.join(
        neurovault_directory(), '.neurovault_metadata.db')
    if not os.path.isfile(db_path):
        try:
            with open(db_path, 'wb'):
                pass
        except EnvironmentError as error:
            if errno.errorcode[error.errno] not in ['EPERM', 'EACCES']:
                raise
            _logger.warning('Could not create database: no write access')
    return db_path


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
    image_id : int
        The Neurovault id of the statistical map.

    target_file : str
        Path to the file in which the terms will be stored on disk
        (a json file).

    temp_dir : str
        Path to directory used by ``_simple_download``.

    Returns
    -------
    None

    """
    query = urljoin(_NEUROSYNTH_FETCH_WORDS_URL,
                    '?neurovault={}'.format(image_id))
    _simple_download(query, target_file, temp_dir)


def neurosynth_words_vectorized(word_files, **kwargs):
    """Load Neurosynth data from disk into an (n files, voc size) matrix

    Neurosynth data is saved on disk as ``{word: weight}``
    dictionaries for each image, this function reads it and returs a
    vocabulary list and a term weight matrix.

    Parameters:
    ----------
    word_files : container
        The paths to the files from which to read word weights (each
        is supposed to contain the Neurosynth response for a
        particular image).

    Keyword arguments are passed on to
    `sklearn.feature_extraction.DictVectorizer.

    Returns:
    -------
    vocabulary : list of str
        A list of all the words encountered in the word files.

    frequencies : numpy.ndarray
        An (n images, vocabulary size) array. Each row corresponds to
        an image, and each column corresponds to a word. The words are
        in the same order as in returned vaule `vocabulary`, so that
        `frequencies[i, j]` corresponds to the weight of
        `vocabulary[j]` for image ``i``.  This matrix is computed by
        an ``sklearn.feature_extraction.DictVectorizer`` instance.

    See Also
    --------
    sklearn.feature_extraction.DictVectorizer

    """
    words = []
    for file_name in word_files:
        try:
            with open(file_name) as word_file:
                info = json.load(word_file)
                words.append(info['data']['values'])
        except Exception as e:
            _logger.warning(
                'could not load words from file {}'.format(file_name))
            words.append({})
    vectorizer = DictVectorizer(**kwargs)
    frequencies = vectorizer.fit_transform(words).toarray()
    vocabulary = np.asarray(vectorizer.feature_names_)
    return frequencies, vocabulary


class BaseDownloadManager(object):
    """Base class for all Neurovault download managers.

    download managers are used as parameters for fetch_neurovault;
    they download the files and store them on disk.

    A ``BaseDownloadManager`` does not download anything, but
    increments a counter and raises a ``MaxImagesReached`` exception
    when the specified max number of images has been reached.

    Subclasses should override ``_collection_hook`` and
    ``_image_hook`` in order to perform the actual work.  They should
    not override ``image`` as it is responsible for stopping the
    stream of metadata when the max numbers of images has been
    reached.

    Parameters
    ----------
    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. This is
        passed on to _get_dataset_dir, which may result in another
        directory being used if the one that was specified is not
        valid.

    max_images : int, optional(default=100)
        Maximum number of images to fetch. ``None`` or a negative
        value means download as many as you can.

    """
    def __init__(self, neurovault_data_dir, max_images=100):
        self.nv_data_dir_ = neurovault_data_dir
        if max_images is not None and max_images < 0:
            max_images = None
        self.max_images_ = max_images
        self.already_downloaded_ = 0
        self.write_ok_ = os.access(self.nv_data_dir_, os.W_OK)

    def collection(self, collection_info):
        """Receive metadata for a collection and take necessary actions.

        The actual work is delegated to ``self._collection_hook``,
        which subclasses should override.

        """
        return self._collection_hook(collection_info)

    def image(self, image_info):
        """Receive metadata for an image and take necessary actions.

        Stop metadata stream if maximum number of images has been
        reached.

        The actual work is delegated to ``self._image_hook``,
        which subclasses should override.

        """
        if self.already_downloaded_ == self.max_images_:
            raise MaxImagesReached()
        image_info = self._image_hook(image_info)
        if image_info is not None:
            self.already_downloaded_ += 1
        return image_info

    def update_image(self, image_info):
        return image_info

    def update_collection(self, collection_info):
        return collection_info

    def update(self, image_info, collection_info):
        """Act when metadata stored on disk is seen again."""
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
        """Prepare for download session."""
        pass

    def finish(self):
        """Cleanup after download session."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.finish()


def _write_metadata(metadata, file_name):
    """Save metadata to disk.

    Absolute paths are not written; they are recomputed using the
    relative paths when data is loaded again, so that if the
    Neurovault directory has been moved paths are still valid.

    Parameters
    ----------
    metadata : dict
        Dictionary representing metadata for a file or a
        collection. Any key containing 'absolute' is ignored.

    file_name : str
        Path to the file in which to write the data.

    """
    metadata = {k: v for k, v in metadata.items()
                if 'absolute' not in k}
    with open(file_name, 'w') as metadata_file:
        json.dump(metadata, metadata_file)


def _add_absolute_paths(root_dir, metadata, force=True):
    """Add absolute paths to a dictionary containing relative paths.

    Parameters
    ----------
    root_dir : str
        The root of the data directory, to prepend to relative paths
        in order to form absolute paths.

    metadata : dict
        Dictionary containing metadata for a file or a collection. Any
        key containing 'relative' is understood to be mapped to a
        relative path and the corresponding absolute path is added to
        the dictionary.

    force : bool, optional (default=True)
        If ``True``, if an absolute path is already present in the
        metadata, it is replaced with the recomputed value. If
        ``False``, already specified absolute paths have priority.

    Returns
    -------
    metadata : dict
        The metadata enriched with absolute paths.

    """
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


def _re_raise(error):
    raise(error)


def _tolerate_failure(error):
    pass


class DownloadManager(BaseDownloadManager):
    """Store maps, metadata, reduced representations and associated words.

    For each collection, this download manager creates a subdirectory
    in the Neurovault directory and stores in it:
        - Metadata for the collection (in .json files).
        - Metadata for the brain maps (in .json files), the brain maps
          (in .nii.gz files).
        - Optionally, the reduced representations of the brain maps
          (in .npy files).
        - Optionally, for each image, the words weights associated to

    Parameters
    ----------
    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. This is
        passed on to _get_dataset_dir, which may result in another
        directory being used if the one that was specified is not
        valid.

    max_images : int, optional(default=100)
        Maximum number of images to fetch. ``None`` or a negative
        value means download as many as you can.

    temp_dir : str or None, optional (default=None)
        Sandbox directory for downloads.  if None, a temporary
        directory is created by ``tempfile.mkdtemp``.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    fetch_reduced_rep : bool, optional (default=True)
        Wether to download the reduced representations from
        Neurovault.

    neurosynth_error_handler :
        Callable, optional (default=_tolerate_failure)
        What to do when words for an image could not be
        retrieved. The default value keeps the image anyway and
        does not raise an error.

    """
    def __init__(self, neurovault_data_dir=None, temp_dir=None,
                 fetch_neurosynth_words=False, fetch_reduced_rep=True,
                 max_images=100, neurosynth_error_handler=_tolerate_failure):

        super(DownloadManager, self).__init__(
            neurovault_data_dir=neurovault_data_dir, max_images=max_images)
        self.suggested_temp_dir_ = temp_dir
        self.temp_dir_ = None
        self.fetch_ns_ = fetch_neurosynth_words
        self.fetch_reduced_rep_ = fetch_reduced_rep
        self.neurosynth_error_handler_ = neurosynth_error_handler

    def _collection_hook(self, collection_info):
        """Create collection subdir and store metadata.

        Parameters
        ----------
        collection_info : dict
            Collection metadata

        Returns
        -------
        collection_info : dict
            Collection metadata, with local path to collection
            subdirectory added to it.

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
        """Get the Neurosynth words for an image and write them to disk.

        If ``self.fetch_ns_ is ``False``, nothing is done.
        Errors that occur when fetching words from Neurosynth are
        handled by ``self.neurosynth_error_handler_``.
        If the corresponding file already exists on disk, the server
        is not queryied again.

        Parameters
        ----------
        image_info : dict
            Image metadata.

        Returns
        -------
        image_info : dict
            Image metadata, with local paths to image, reduced
            representation (if fetched), and Neurosynth words (if
            fetched) added to it.

        """
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
            if not os.path.isfile(ns_words_absolute_path):
                try:
                    _fetch_neurosynth_words(
                        image_info['id'],
                        ns_words_absolute_path, self.temp_dir_)
                except(URLError, ValueError) as e:
                    _logger.exception(
                        'could not fetch words for image {}'.format(
                            image_info['id']))
                    self.neurosynth_error_handler_(e)
                    return
            image_info[
                'neurosynth_words_relative_path'] = ns_words_relative_path
            image_info[
                'neurosynth_words_absolute_path'] = ns_words_absolute_path
        return image_info

    def _image_hook(self, image_info):
        """Download image, reduced representation, Neurosynth words.

        Wether reduced representation and Neurosynth words are
        downloaded depends on ``self.fetch_reduced_rep_`` and
        ``self.fetch_ns_``.

        Parameters
        ----------
        image_info: dict
            Image metadata.

        Returns
        -------
        image_info: dict
            Image metadata, with local path to image, local path to
            reduced representation (if reduced representation
            available and ``self.fetch_reduced_rep_``), and local path
            to Neurosynth words (if ``self.fetch_ns_``) added to it.

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
        if self.fetch_reduced_rep_ and reduced_image_url is not None:
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
        _logger.info('already fetched {} image{}'.format(
            self.already_downloaded_ + 1,
            ('s' if self.already_downloaded_ + 1 > 1 else '')))
        return image_info

    def update_image(self, image_info):
        """Download Neurosynth words if necessary.

        If ``self.fetch_ns_`` is set and Neurosynth words are not on
        disk, fetch them and add their location to image metadata.

        """
        if not self.write_ok_:
            return image_info
        image_info = self._add_words(image_info)
        metadata_file_path = os.path.join(
            os.path.dirname(image_info['absolute_path']),
            'image_{}_metadata.json'.format(image_info['id']))
        _write_metadata(image_info, metadata_file_path)
        return image_info

    def start(self):
        """Prepare for a download session.

        If we don't have a sandbox directory for downloads, create
        one.

        """
        if self.temp_dir_ is None:
            self.temp_dir_ = _get_temp_dir(self.suggested_temp_dir_)

    def finish(self):
        """Cleanup after downlaod session.

        If ``self.start`` created a temporary directory for the
        download session, remove it.

        """
        if self.temp_dir_ is None:
            return
        if self.temp_dir_ != self.suggested_temp_dir_:
            shutil.rmtree(self.temp_dir_)
            self.temp_dir_ = None


class SQLiteDownloadManager(DownloadManager):
    """Store Neurovault data; store metadata in an sqlite database.

    All data and metadata is stored as by DownloadManager instances,
    and (a subset of) the metadata is stored in an sqlite database so
    that it can be accessed more easily.

    Parameters
    ----------
    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. This is
        passed on to _get_dataset_dir, which may result in another
        directory being used if the one that was specified is not
        valid.

    max_images : int, optional(default=100)
        Maximum number of images to fetch. ``None`` or a negative
        value means download as many as you can.

    temp_dir : str or None, optional (default=None)
        Sandbox directory for downloads.  if None, a temporary
        directory is created by ``tempfile.mkdtemp``.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    fetch_reduced_rep : bool, optional (default=True)
        Wether to download the reduced representations from
        Neurovault.

    neurosynth_error_handler :
        Callable, optional (default=_tolerate_failure)
        What to do when words for an image could not be
        retrieved. The default value keeps the image anyway and
        does not raise an error.

    image_fields : Container, optional
        (default=_IMAGE_BASIC_FIELDS_SQL.keys())
        Fields of the image metadata to include in sqlite database.

    collection_fields : Container, optional
        (default=_COLLECTION_BASIC_FIELDS_SQL.keys())
        Fields of the image metadata to include in sqlite database.

    """
    def __init__(self, image_fields=_IMAGE_BASIC_FIELDS_SQL.keys(),
                 collection_fields=_COLLECTION_BASIC_FIELDS_SQL.keys(),
                 **kwargs):
        super(SQLiteDownloadManager, self).__init__(**kwargs)
        self.connection_ = None
        self.cursor_ = None
        self.im_fields_ = _filter_field_names(image_fields,
                                              _ALL_IMAGE_FIELDS_SQL)
        self.col_fields_ = _filter_field_names(collection_fields,
                                               _ALL_COLLECTION_FIELDS_SQL)
        self._update_sql_statements()
        self.write_db_ok_ = self.write_ok_ and os.access(
            neurovault_metadata_db_path(), os.W_OK)

    def _update_sql_statements(self):
        """Prepare SQL statements used to store metadata."""
        self.im_insert_ = _get_insert_string('images', self.im_fields_)
        self.col_insert_ = _get_insert_string('collections', self.col_fields_)
        self.im_update_ = _get_update_string('images', self.im_fields_)
        self.col_update_ = _get_update_string('collections', self.col_fields_)

    def _add_to_collections(self, collection_info):
        """Add metadata for a collection to 'collections' table

        Parameters
        ----------
        collection_info : dict
            Collection metadata

        Returns
        -------
        collection_info : dict
            Identical to the argument `collection_info`.

        """
        values = [collection_info.get(field) for field in self.col_fields_]
        try:
            self.cursor_.execute(self.col_insert_, values)
        except sqlite3.IntegrityError:
            self.cursor_.execute(self.col_update_, values)
        return collection_info

    def _collection_hook(self, collection_info):
        """Create collection subdir and store metadata.

        Parameters
        ----------
        collection_info : dict
            Collection metadata

        Returns
        -------
        collection_info : dict
            Collection metadata, with local path to collection
            subdirectory added to it.

        """
        collection_info = super(SQLiteDownloadManager, self)._collection_hook(
            collection_info)
        collection_info = self._add_to_collections(collection_info)
        return collection_info

    def _add_to_images(self, image_info):
        """Add metadata for an image to 'images' table

        Parameters
        ----------
        image_info : dict
            Image metadata

        Returns
        -------
        image_info : dict
            Identical to the argument `image_info`.

        """
        values = [image_info.get(field) for field in self.im_fields_]
        try:
            self.cursor_.execute(self.im_insert_, values)
        except sqlite3.IntegrityError:
            self.cursor_.execute(self.im_update_, values)
        return image_info

    def _image_hook(self, image_info):
        """Download image, reduced representation, Neurosynth words.

        Wether reduced representation and Neurosynth words are
        downloaded depends on ``self.fetch_reduced_rep_`` and
        ``self.fetch_ns_``.

        Parameters
        ----------
        image_info: dict
            Image metadata.

        Returns
        -------
        image_info: dict
            Image metadata, with local path to image, local path to
            reduced representation (if reduced representation
            available and ``self.fetch_reduced_rep_``), and local path
            to Neurosynth words (if ``self.fetch_ns_``) added to it.

        """
        image_info = super(SQLiteDownloadManager, self)._image_hook(
            image_info)
        image_info = self._add_to_images(image_info)
        return image_info

    def update_image(self, image_info):
        """Update database content for an image.

        If ``self.fetch_ns_`` is set and Neurosynth words are not on
        disk, fetch them and add their location to image metadata.

        """
        if not self.write_db_ok_:
            return image_info
        super(SQLiteDownloadManager, self).update_image(image_info)
        return self._add_to_images(image_info)

    def update_collection(self, collection_info):
        if not self.write_db_ok_:
            return collection_info
        """Update database content for a collection."""
        super(SQLiteDownloadManager, self).update_collection(collection_info)
        return self._add_to_collections(collection_info)

    def start(self):
        """Prepare for a download session.

        A connection to the local Neurovault database is open and
        columns are added to its tables if necessary.

        See Also
        --------
        SQLiteDownloadManager._update_schema

        """
        super(SQLiteDownloadManager, self).start()
        _logger.debug('starting download manager')
        self.connection_ = local_database_connection()
        self.cursor_ = local_database_cursor()
        self._update_schema()

    def _update_schema(self):
        """Create or alter a database so it contains the required tables.

        If a database already exists, the required columns
        (``self.im_fields_`` and ``self.col_fields_``) are added to
        its tables if absent. Existing columns are not dropped and
        will also be filled during the download session. If no
        database exists, it is created.

        """
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
        """Cleanup after a download session.

        Commit changes and close database connection.

        """
        super(SQLiteDownloadManager, self).finish()
        if self.connection_ is None:
            return
        close_database_connection(_logger.debug)
        self.connection_ = None


def _scroll_collection(collection, image_terms, image_filter, batch_size,
                       download_manager,
                       previous_consecutive_fails, max_consecutive_fails):
    """Iterate over the content of a collection on Neurovault server.

    Parameters
    ----------
    collection : dict
        The collection metadata.

    image_terms : dict
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be ignored.

    image_filter : Callable
        Images for which `image_filter(image_metadata)` is ``False``
        will be ignored.

    batch_size : int
        Neurovault sends metadata in batches. `batch_size` is the size
        of the batches to ask for.

    previous_consecutive_fails : int
        How many images have failed to be downloaded since last
        successful download.

    max_consecutive_fails : int
        If more than `max_consecutive_fails` images in a row fail to
        be downloaded, we consider there is a problem and stop the
        download session.

    Yields
    ------
    image : dict
        Metadata for an image.

    consecutive_fails : int
        How many images have failed to be downloaded since last
        successful download.

    Raises
    ------
    MaxImagesReached
        If enough images have been downloaded.

    RuntimeError
        If more than `max_consecutive_fails` images have failed in a
        row.

    """
    n_im_in_collection = 0
    consecutive_fails = previous_consecutive_fails
    query = urljoin(_NEUROVAULT_COLLECTIONS_URL,
                    '{}/images/'.format(collection['id']))
    images = _scroll_server_results(
        query, query_terms=image_terms,
        local_filter=image_filter,
        prefix_msg='scroll images from collection {}: '.format(
            collection['id']),
        batch_size=batch_size)
    for image in images:
        try:
            image = download_manager.image(image)
            consecutive_fails = 0
            yield image, consecutive_fails
            n_im_in_collection += 1
        except MaxImagesReached:
            raise
        except Exception:
            consecutive_fails += 1
            _logger.exception(
                '_scroll_collection: bad image: {}'.format(image))
        if consecutive_fails == max_consecutive_fails:
            _logger.error(
                '_scroll_server_data stopping after {} bad images'.format(
                    consecutive_fails))
            raise RuntimeError(
                '{} consecutive bad images'.format(consecutive_fails))

    _logger.info(
        'on neurovault.org: '
        '{} image{} matched query in collection {}'.format(
            (n_im_in_collection if n_im_in_collection else 'no'),
            ('s' if n_im_in_collection > 1 else ''), collection['id']))


# TODO: finish docstring.
def _scroll_server_data(collection_query_terms={},
                        collection_local_filter=_empty_filter,
                        image_query_terms={}, image_local_filter=_empty_filter,
                        download_manager=None, max_images=None,
                        metadata_batch_size=None, max_consecutive_fails=5):
    """Iterate over neurovault.org results for a query.

    Parameters
    ----------
    collection_query_terms : dict
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True``
        for every key, value pair will be ignored.

    collection_local_filter : Callable
        Collections for which
        `collection_local_filter(collection_metadata)` is ``False``
        will be ignored.

    image_query_terms : dict
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be ignored.

    image_local_filter : Callable
        Images for which `image_local_filter(image_metadata)` is
        ``False`` will be ignored.

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.
        If None, one is constructed.

    max_images : int, optional (default=None)
        Maximum number of images to download; only used if
        `download_manager` is None.

    metadata_batch_size : int, optional(default=None)
        Neurovault sends metadata in batches. `batch_size` is the size
        of the batches to ask for. If ``None``, the default
        ``_DEFAULT_BATCH_SIZE`` will be used.

    max_consecutive_fails : int
        If more than `max_consecutive_fails` images in a row fail to
        be downloaded, we consider there is a problem and stop the
        download session.

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the image's collection.

    Raises
    ------
    MaxImagesReached
        If enough images have been downloaded.

    RuntimeError
        If more than `max_consecutive_fails` images have failed in a
        row.

    """
    if download_manager is None:
        download_manager = BaseDownloadManager(
            neurovault_data_dir=neurovault_directory(), max_images=max_images)

    collections = _scroll_server_results(
        _NEUROVAULT_COLLECTIONS_URL, query_terms=collection_query_terms,
        local_filter=collection_local_filter,
        prefix_msg='scroll collections: ', batch_size=metadata_batch_size)
    consecutive_fails = 0

    with download_manager:
        for collection in collections:
            try:
                collection = download_manager.collection(collection)
            except MaxImagesReached:
                raise
            except Exception:
                _logger.exception(
                    '_scroll_server_data: bad collection: {}'.format(
                        collection))
                raise
            collection_content = _scroll_collection(
                collection, image_query_terms, image_local_filter,
                metadata_batch_size, download_manager,
                consecutive_fails, max_consecutive_fails)
            while True:
                try:
                    image, consecutive_fails = next(collection_content)
                except MaxImagesReached:
                    raise
                except StopIteration:
                    break
                yield image, collection


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
    """Load a json file and add image, reduced rep and words paths"""
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
    """Iterate over local Neurovault data matching a query.

    Parameters
    ----------
    neurovault_dir : str
        Path to Neurovault data directory.

    collection_filter : Callable
        Collections for which
        `collection_local_filter(collection_metadata)` is ``False``
        will be ignored.

    image_filter : Callable
        Images for which `image_local_filter(image_metadata)` is
        ``False`` will be ignored.

    max_images : int, optional (default=None)
        Maximum number of images' metadata to load

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the image's collection.

    """
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
    server_terms = {k: terms_.pop(k) for k in
                    available_on_server.intersection(terms_.keys()) if
                    isinstance(terms_[k], str) or isinstance(terms_[k], int)}
    return terms_, server_terms


def _move_unknown_terms_to_local_filter(terms, local_filter,
                                        available_on_server):
    """Move filters handled by the server inside URL.

    Some filters are available on the server and can be inserted into
    the URL query. The rest will have to be applied on metadata
    locally.

    """
    local_terms, server_terms = _split_terms(terms, available_on_server)
    local_filter = ResultFilter(query_terms=local_terms).AND(local_filter)
    return server_terms, local_filter


def _prepare_local_scroller(neurovault_dir, collection_terms,
                            collection_filter, image_terms,
                            image_filter, max_images):
    """Construct filters for call to ``_scroll_local_data``."""
    collection_local_filter = ResultFilter(
        **collection_terms).AND(collection_filter)
    image_local_filter = ResultFilter(**image_terms).AND(image_filter)
    local_data = _scroll_local_data(
        neurovault_dir, collection_filter=collection_local_filter,
        image_filter=image_local_filter, max_images=max_images)

    return local_data


# TODO: finish docstring
def _prepare_remote_scroller(collection_terms, collection_filter,
                             image_terms, image_filter,
                             collection_ids, image_ids,
                             download_manager, max_images):
    """Construct filters for call to ``_scroll_server_data``."""
    collection_terms, collection_filter = _move_unknown_terms_to_local_filter(
        collection_terms, collection_filter,
        _COL_FILTERS_AVAILABLE_ON_SERVER)

    collection_filter = ResultFilter(
        id=NotIn(collection_ids)).AND(collection_filter)

    image_terms, image_filter = _move_unknown_terms_to_local_filter(
        image_terms, image_filter,
        _IM_FILTERS_AVAILABLE_ON_SERVER)

    image_filter = ResultFilter(id=NotIn(image_ids)).AND(image_filter)

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


class _EmptyContext(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


# TODO: finish docstring
def _join_local_and_remote(neurovault_dir, mode='download_new',
                           collection_terms={},
                           collection_filter=_empty_filter,
                           image_terms={}, image_filter=_empty_filter,
                           download_manager=None, max_images=None):
    """Iterate over results from disk, then those found on neurovault.org

    Parameters
    ----------
    neurovault_dir : str
        Path to Neurovault data directory.

    mode : {'download_new', 'overwrite', 'offline'}
        - 'download_new' (the default) means download only files that
          are not already on disk.
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    collection_terms : dict, optional (default={})
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True``
        for every key, value pair will be ignored.

    collection_filter : Callable, optional (default=_empty_filter)
        Collections for which
        `collection_local_filter(collection_metadata)` is ``False``
        will be ignored.

    image_terms : dict, optional (default={})
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be ignored.

    image_filter : Callable, optional (default=_empty_filter)
        Images for which `image_local_filter(image_metadata)` is
        ``False`` will be ignored.

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.
        If None, one is constructed if required (i.e. we are not
        working offline).

    max_images : int, optional (default=None)
        Maximum number of images to download; only used if
        `download_manager` is None.

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the image's collection.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    Tries to yield `max_images` images; stops early if we have fetched
    all the images matching the filters or if an uncaught exception is
    raised during download

    """
    mode = mode.lower()
    if mode not in ['overwrite', 'download_new', 'offline']:
        raise ValueError(
            'supported modes are overwrite,'
            ' download_new, offline; got {}'.format(mode))
    image_ids, collection_ids = set(), set()
    if mode == 'overwrite':
        local_data = tuple()
    else:
        _logger.debug('reading local neurovault data')
        local_data = _prepare_local_scroller(
            neurovault_dir, collection_terms, collection_filter,
            image_terms, image_filter, max_images)
        context = (download_manager if download_manager is not None
                   else _EmptyContext())
        update = (download_manager.update if download_manager is not None
                  else _return_same)
        with context:
            for image, collection in local_data:
                image, collection = update(image, collection)
                image_ids.add(image['id'])
                collection_ids.add(collection['id'])
                yield image, collection

    if mode == 'offline':
        return
    if max_images is not None and len(image_ids) >= max_images:
        return

    _logger.debug('reading server neurovault data')
    server_data = _prepare_remote_scroller(collection_terms, collection_filter,
                                           image_terms, image_filter,
                                           collection_ids, image_ids,
                                           download_manager, max_images)
    while True:
        try:
            image, collection = next(server_data)
        except StopIteration:
            return
        except Exception:
            _logger.exception('downloading data from server stopped early')
            _logger.error('downloading data from server stopped early: '
                          'see stacktrace above')
            return
        yield image, collection


def basic_collection_terms():
    """Return a term filter that excludes empty collections."""
    return {'number_of_images': NotNull(),
            'id': NotIn(_KNOWN_BAD_COLLECTION_IDS)}


def basic_image_terms():
    """Filter that selects unthresholded F, T and Z maps in mni space"""
    return {'not_mni': False, 'is_valid': True, 'is_thresholded': False,
            'map_type': IsIn({'F map', 'T map', 'Z map'}),
            'id': NotIn(_KNOWN_BAD_IMAGE_IDS)}


def _move_col_id(im_terms, col_terms):
    """Reposition 'collection_id' term.

    If the collection id was specified in image filters, move it to
    the collection filters for efficiency.

    This makes specifying the collection id as a keyword argument for
    fetch_neurovault efficient.

    """
    if 'collection_id' in im_terms:
        if 'id' not in col_terms:
            col_terms['id'] = im_terms.pop('collection_id')
        elif col_terms['id'] == im_terms['collection_id']:
            im_terms.pop('collection_id')
        else:
            warnings.warn('You specified contradictory collection ids, '
                          'one in the image filters and one in the '
                          'collection filters')
    return im_terms, col_terms


# TODO: finish docstring
def fetch_neurovault(max_images=100,
                     collection_terms=basic_collection_terms(),
                     collection_filter=_empty_filter,
                     image_terms=basic_image_terms(),
                     image_filter=_empty_filter,
                     mode='download_new',
                     neurovault_data_dir=None,
                     fetch_neurosynth_words=False,
                     download_manager=None, **kwargs):
    """Download data from neurovault.org and neurosynth.org.

    Parameters
    ----------
    max_images : int, optional (default=None)
        Maximum number of images to fetch.

    collection_terms : dict, optional (default=basic_collection_terms())
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True``
        for every key, value pair will be ignored.

    collection_filter : Callable, optional (default=_empty_filter)
        Collections for which
        `collection_local_filter(collection_metadata)` is ``False``
        will be ignored.

    image_terms : dict, optional (default=basic_image_terms())
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be ignored.

    image_filter : Callable, optional (default=_empty_filter)
        Images for which `image_local_filter(image_metadata)` is
        ``False`` will be ignored.

    mode : {'download_new', 'overwrite', 'offline'}
        - 'download_new' (the default) means download only files that
          are not already on disk.
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    neurovault_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. Another
        directory may be used if the one that was specified is not
        valid.

    neurovault_data_dir : str
        Path to Neurovault data directory.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.
        If None, one is constructed (an SQLiteDownloadManager).

    Keyword arguments are understood to be filter terms for images, so
    for example ``map_type='Z map'`` means only download Z-maps;
    ``collection_id=35`` means download images from collection 35
    only.

    Returns
    -------
    Bunch
        A dict-like object which exposes its items as attributes.  It
        contains:
            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
            dictionaries.
            - 'collections_meta', the metadata for the
            collections.

        If `fetch_neurosynth_words` was set, it also
        contains:
            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
            neurosynth.org for each image, such that the weight of word
            `vocabulary[j]` for the image found in `images[i]` is
            `word_frequencies[i, j]`

    See Also
    --------
    basic_image_terms
        The terms on which images are filtered by default.

    basic_collection_terms
        The terms on which collections are filtered by default.

    DownloadManager, SQLiteDownloadManager
        Possible handlers for the downloaded data.

    Some authors have included many fields in the metadata they
    provide; in order to make it easier to figure out which fields are
    used by most authors and which are interesting to you, these
    functions could be of help:

    plot_fields_occurrences
        Show a bar plot of how many images (resp collections) use a
        particular image (resp collection) metadata field.

    show_neurovault_image_keys, show_neurovault_collection_keys
        Show the field names that were seen in metadata and the types
        of the values that were associated to them. For this
        information, you can also have a look at the module-level
        variables _IMAGE_BASIC_FIELDS, _COLLECTION_BASIC_FIELDS,
        _ALL_COLLECTION_FIELDS and _ALL_IMAGE_FIELDS.

    Notes
    -----
    The default behaviour is to store the most important fields (which
    you can define) of metadata in an ``sqlite`` database, which is
    actually just a file but can be queried like an SQL database. So
    in addition to the ``Bunch`` returned by this function, if you
    find it more convenient, you can access the data through this
    other interface.

    Images and collections from disk are fetched before remote data.

    Tries to yield `max_images` images; stops early if we have fetched
    all the images matching the filters or if an uncaught exception is
    raised during download

    References
    ----------
    [1] Gorgolewski KJ, Varoquaux G, Rivera G, Schwartz Y, Ghosh SS,
        Maumet C, Sochat VV, Nichols TE, Poldrack RA, Poline J-B,
        Yarkoni T and Margulies DS (2015) NeuroVault.org: a web-based
        repository for collecting and sharing unthresholded
        statistical maps of the human brain. Front. Neuroinform. 9:8.
        doi: 10.3389/fninf.2015.00008

    """
    image_terms = dict(image_terms, **kwargs)
    image_terms, collection_terms = _move_col_id(image_terms, collection_terms)

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
        neurovault_dir=neurovault_data_dir, mode=mode,
        collection_terms=collection_terms, collection_filter=collection_filter,
        image_terms=image_terms, image_filter=image_filter,
        download_manager=download_manager, max_images=max_images)

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
             [meta.get('neurosynth_words_absolute_path') for
              meta in images_meta])
    return result


def refresh_db(**kwargs):
    """Update local database with metadata cached in json files.

    This is mostly called automatically so that the database is always
    up-to-date, but it can be used by a user to add columns to tables.

    See Also
    --------
    SQLiteDownloadManager

    """
    if not os.access(neurovault_metadata_db_path(), os.W_OK):
        return
    _logger.debug('refreshing local database')
    download_manager = SQLiteDownloadManager(
        neurovault_data_dir=neurovault_directory(), **kwargs)
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
        stop after seeing metadata for max_images images.  If None,
        read metadata for all images and collections.

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
    meta = getattr(_get_all_neurovault_keys, 'meta_', None)

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
    """Display keys found in Neurovault metadata for images.

    The results are displayed as many lines of the form:

    field_name: (type, number of images that have filled this field)

    Parameters
    ----------
    max_images: int, optional (default=None)
        stop after seeing metadata for max_images images.  If None,
        read metadata for all images and collections.

    Returns
    -------
    None

    """
    pprint(_get_all_neurovault_keys(max_images)[0])


def show_neurovault_collection_keys(max_images=300):
    """Display keys found in Neurovault metadata for collections.

    The results are displayed as many lines of the form:

    field_name: (type, number of collections that have filled this field)

    Parameters
    ----------
    max_images: int, optional (default=None)
        stop after seeing metadata for max_images images.  If None,
        read metadata for all images and collections.

    Returns
    -------
    None

    """
    pprint(_get_all_neurovault_keys(max_images)[1])


def _which_keys_are_unused(max_images=None):
    """Find which metadata fields are never filled.

    Parameters
    ----------
    max_images: int, optional (default=None)
        stop after seeing metadata for max_images images.  If None,
        read metadata for all images and collections.

    Returns
    -------
    im_unused, coll_unused
        ``set`` objects of field names which are unused.

    """
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
    """Helper function for ``plot_fields_occurrences``"""
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
    """Helper function for ``plot_fields_occurrences``"""
    gs_im = GridSpec(1, 1, bottom=.65, top=.95)
    gs_col = GridSpec(1, 1, bottom=.2, top=.5)
    ax_im = plt.subplot(gs_im[:])
    ax_im.set_title('image fields')
    ax_col = plt.subplot(gs_col[:])
    ax_col.set_title('column fields', fontsize='xx-large')
    return ax_im, ax_col


def plot_fields_occurrences(max_images=300, **kwargs):
    """Draw a histogram of how often metadata fields are filled."""
    all_keys = _get_all_neurovault_keys(max_images)
    axis_arr = _prepare_subplots_fields_occurrences()
    for table, ax in zip(all_keys, axis_arr):
        _fields_occurences_bar(table, ax=ax, **kwargs)


def _filter_field_names(required_fields, ref_fields):
    """Keep the fields that are present in a reference set.

    Used to select only known fields, find the type that is associated
    to them, and control what can be inserted in an SQL statement.

    """
    filtered = OrderedDict()
    for field_name in required_fields:
        if field_name in ref_fields:
            filtered[field_name] = ref_fields[field_name]
        else:
            _logger.warning(
                'rejecting unknown column name: {}'.format(field_name))
    return filtered


def _get_columns_string(required_fields, ref_fields):
    """Prepare a string describing columns for an SQL table.

    Only fields present in `ref_fields` are accepted; only elements of
    a predetermined set of strings are inserted in this string.

    """
    fields = ['{} {}'.format(n, v) for
              n, v in _filter_field_names(required_fields, ref_fields).items()]
    return ', '.join(fields)


def _get_insert_string(table_name, fields):
    """Prepare an SQL INSERT INTO statement."""
    return "INSERT INTO {} ({}) VALUES ({})".format(
        table_name,
        ', '.join(fields),
        ('?, ' * len(fields))[:-2])


def _get_update_string(table_name, fields):
    """Prepare an SQL UPDATE statement."""
    set_str = ','.join(["{}=:{}".format(field, field) for field in fields])
    return "UPDATE {} SET {} WHERE id=:id".format(table_name, set_str)


def _table_exists(cursor, table_name):
    cursor.execute("SELECT * FROM sqlite_master WHERE name=?", (table_name,))
    return bool(cursor.fetchall())


def local_database_connection():
    """Get access to the local sqlite database holding Neurovault metadata.

    This is for users who find SQL syntax more convenient than
    manipulating python dicts. It can also be useful to users who also
    use ``pandas``, as they can very easily load Neurovault metadata
    into a ``pandas.DataFrame`` object:

    df = pd.read_sql_query('SELECT * FROM images', local_database_connection())

    """
    if getattr(local_database_connection, 'connection_', None) is not None:
        return local_database_connection.connection_
    db_path = neurovault_metadata_db_path()
    local_database_connection.connection_ = sqlite3.connect(db_path)
    local_database_connection.connection_.row_factory = sqlite3.Row
    return local_database_connection.connection_


def local_database_cursor():
    return local_database_connection().cursor()


@atexit.register
def close_database_connection(log_fun=_logger.info):
    """Commit changes and close local database if necessary."""
    try:
        local_database_connection.connection_.commit()
        local_database_connection.connection_.close()
        log_fun(
            'committed changes to local database and closed connection')
    except (AttributeError, sqlite3.ProgrammingError):
        pass
    except Exception as e:
        _logger.exception()
    local_database_connection.connection_ = None


def _create_schema(cursor, im_fields=_IMAGE_BASIC_FIELDS,
                   col_fields=_COLLECTION_BASIC_FIELDS):
    """Create images and collections tables in an sqlite database.

    Only elements from _ALL_COLLECTION_FIELDS_SQL and
    _ALL_IMAGE_FIELDS_SQL will actually be used.

    Parameters:
    ----------
    cursor : sqlite3.Cursor
        Cursor for the database.

    im_fields : Container, optional (default=_IMAGE_BASIC_FIELDS)
        Columns to include in images table.

    col_fields : Container, optional (default=_COLLECTION_BASIC_FIELDS)
        Columns to include in collections table.

    """
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
    """Find out about the columns of a table and the type affinities.

    Also returns (part of) the statement used to create the table.

    """
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
    """Return the column names and their type affinities for a table."""
    columns = table_info(cursor, table_name)[1]
    if columns is None:
        return None
    return next(zip(*columns))


def read_sql_query(query, bindings=(), as_columns=True, curs=None):
    """Get response from local Neurovault database for an SQL query.

    Parameters
    ----------
    query : str
        The query (may include place holders for parameter
        substitution).

    bindings : tuple or dict, optional (default=())
        The bindings for the place holders, if any were used in the
        query (tuple if question mark style, dict if named style; see
        ``sqlite3`` documentation).

    as_columns: bool, optional (default=True)
        If ``False``, return the result as a list of ``sqlite3.Row``
        objects (can be indexed with indices, or as dictionaries with
        the column names, see sqlite3 doc.)
        If ``True``, transpose the result and return it as an ordered
        dictionary of columns. In this case each key in the dictionary
        is a column name (or alias if specified in the query), and the
        corresponding value is a one-dimensional numpy array.

    Returns
    -------
    response : OrderedDict or list
        The result of the query, as a dictionary of columns or a list
        of rows.

    See Also
    --------
    sqlite3

    Examples
    --------
    >>> data = read_sql_query('SELECT images.id AS image_id, '
                              'images.absolute_path AS image_path, '
                              'collections.id AS collection_id, '
                              'collections.DOI FROM images '
                              'INNER JOIN collections ON '
                              'images.collection_id=collections.id')

    >>> print(list(data.keys()))

    """
    if curs is None:
        curs = local_database_cursor()
    curs.execute(query, bindings)
    resp = curs.fetchall()
    if not resp:
        col_names = list(zip(*curs.description))[0]
        return OrderedDict([(name, []) for name in col_names])
    if not as_columns:
        return resp
    col_names = resp[0].keys()
    cols = zip(*resp)
    cols = map(np.asarray, cols)
    response = OrderedDict(zip(col_names, cols))
    if 'neurosynth_words_absolute_path' in query:
        frequencies, vocabulary = neurosynth_words_vectorized(
            response['neurosynth_words_absolute_path'])
        response['word_frequencies'] = frequencies
        response['vocabulary'] = vocabulary
    return response
