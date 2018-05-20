#
# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from oslo_policy import policy

from nova.policies import base


BASE_POLICY_NAME = 'os_compute_api:wrs-providernet'


wrs_providernet_policies = [
    policy.DocumentedRuleDefault(
        BASE_POLICY_NAME,
        base.RULE_ADMIN_OR_OWNER,
        """Show PCI usage aggregated values for a provider network.""",
        [
            {
                'method': 'GET',
                'path': '/wrs-providernet/{network_id}'
           }
        ]),
]


def list_rules():
    return wrs_providernet_policies
