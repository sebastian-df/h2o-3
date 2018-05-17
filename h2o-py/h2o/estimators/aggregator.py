#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# This file is auto-generated by h2o-3/h2o-bindings/bin/gen_python.py
# Copyright 2016 H2O.ai;  Apache License Version 2.0 (see LICENSE for details)
#
from __future__ import absolute_import, division, unicode_literals

from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric


class H2OAggregatorEstimator(H2OEstimator):
    """
    Aggregator

    """

    algo = "aggregator"

    def __init__(self, **kwargs):
        super(H2OAggregatorEstimator, self).__init__()
        self._parms = {}
        names_list = {"model_id", "training_frame", "response_column", "ignored_columns", "ignore_const_cols",
                      "target_num_exemplars", "rel_tol_num_exemplars", "transform", "categorical_encoding",
                      "save_mapping_frame", "num_iteration_without_new_exemplar"}
        if "Lambda" in kwargs: kwargs["lambda_"] = kwargs.pop("Lambda")
        for pname, pvalue in kwargs.items():
            if pname == 'model_id':
                self._id = pvalue
                self._parms["model_id"] = pvalue
            elif pname in names_list:
                # Using setattr(...) will invoke type-checking of the arguments
                setattr(self, pname, pvalue)
            else:
                raise H2OValueError("Unknown parameter %s = %r" % (pname, pvalue))
        self._parms["_rest_version"] = 99

    @property
    def training_frame(self):
        """
        Id of the training data frame.

        Type: ``H2OFrame``.
        """
        return self._parms.get("training_frame")

    @training_frame.setter
    def training_frame(self, training_frame):
        assert_is_type(training_frame, None, H2OFrame)
        self._parms["training_frame"] = training_frame


    @property
    def response_column(self):
        """
        Response variable column.

        Type: ``str``.
        """
        return self._parms.get("response_column")

    @response_column.setter
    def response_column(self, response_column):
        assert_is_type(response_column, None, str)
        self._parms["response_column"] = response_column


    @property
    def ignored_columns(self):
        """
        Names of columns to ignore for training.

        Type: ``List[str]``.
        """
        return self._parms.get("ignored_columns")

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        assert_is_type(ignored_columns, None, [str])
        self._parms["ignored_columns"] = ignored_columns


    @property
    def ignore_const_cols(self):
        """
        Ignore constant columns.

        Type: ``bool``  (default: ``True``).
        """
        return self._parms.get("ignore_const_cols")

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        assert_is_type(ignore_const_cols, None, bool)
        self._parms["ignore_const_cols"] = ignore_const_cols


    @property
    def target_num_exemplars(self):
        """
        Targeted number of exemplars

        Type: ``int``  (default: ``5000``).
        """
        return self._parms.get("target_num_exemplars")

    @target_num_exemplars.setter
    def target_num_exemplars(self, target_num_exemplars):
        assert_is_type(target_num_exemplars, None, int)
        self._parms["target_num_exemplars"] = target_num_exemplars


    @property
    def rel_tol_num_exemplars(self):
        """
        Relative tolerance for number of exemplars (e.g, 0.5 is +/- 50 percents)

        Type: ``float``  (default: ``0.5``).
        """
        return self._parms.get("rel_tol_num_exemplars")

    @rel_tol_num_exemplars.setter
    def rel_tol_num_exemplars(self, rel_tol_num_exemplars):
        assert_is_type(rel_tol_num_exemplars, None, numeric)
        self._parms["rel_tol_num_exemplars"] = rel_tol_num_exemplars


    @property
    def transform(self):
        """
        Transformation of training data

        One of: ``"none"``, ``"standardize"``, ``"normalize"``, ``"demean"``, ``"descale"``  (default: ``"normalize"``).
        """
        return self._parms.get("transform")

    @transform.setter
    def transform(self, transform):
        assert_is_type(transform, None, Enum("none", "standardize", "normalize", "demean", "descale"))
        self._parms["transform"] = transform


    @property
    def categorical_encoding(self):
        """
        Encoding scheme for categorical features

        One of: ``"auto"``, ``"enum"``, ``"one_hot_internal"``, ``"one_hot_explicit"``, ``"binary"``, ``"eigen"``,
        ``"label_encoder"``, ``"sort_by_response"``, ``"enum_limited"``  (default: ``"auto"``).
        """
        return self._parms.get("categorical_encoding")

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        assert_is_type(categorical_encoding, None, Enum("auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder", "sort_by_response", "enum_limited"))
        self._parms["categorical_encoding"] = categorical_encoding


    @property
    def save_mapping_frame(self):
        """
        Whether to export the mapping of the aggregated frame

        Type: ``bool``  (default: ``False``).
        """
        return self._parms.get("save_mapping_frame")

    @save_mapping_frame.setter
    def save_mapping_frame(self, save_mapping_frame):
        assert_is_type(save_mapping_frame, None, bool)
        self._parms["save_mapping_frame"] = save_mapping_frame


    @property
    def num_iteration_without_new_exemplar(self):
        """
        The number of iterations to run before aggregator exits if the number of exemplars collected didn't change

        Type: ``int``  (default: ``500``).
        """
        return self._parms.get("num_iteration_without_new_exemplar")

    @num_iteration_without_new_exemplar.setter
    def num_iteration_without_new_exemplar(self, num_iteration_without_new_exemplar):
        assert_is_type(num_iteration_without_new_exemplar, None, int)
        self._parms["num_iteration_without_new_exemplar"] = num_iteration_without_new_exemplar



    @property
    def aggregated_frame(self):
        if (self._model_json is not None and
            self._model_json.get("output", {}).get("output_frame", {}).get("name") is not None):
            out_frame_name = self._model_json["output"]["output_frame"]["name"]
            return H2OFrame.get_frame(out_frame_name)
