#
# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from oslo_policy import policy

from nova.policies import base


BASE_POLICY_NAME = 'os_compute_api:wrs-if'


wrs_if_policies = [
     policy.DocumentedRuleDefault(
         BASE_POLICY_NAME,
         base.RULE_ADMIN_OR_OWNER,
         """Add wrs-if:nics attribute in the server response.

This check is performed only after the check
'os_compute_api:servers:show' for GET /servers/{id} and
'os_compute_api:servers:detail' for GET /servers/detail passes""",


         [
             {
                 'method': 'GET',
                 'path': '/servers/{id}'
             },
             {
                 'method': 'GET',
                 'path': '/servers/detail'
             }
         ]),
]


def list_rules():
    return wrs_if_policies
