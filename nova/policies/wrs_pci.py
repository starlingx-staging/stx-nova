#
# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from oslo_policy import policy

from nova.policies import base


BASE_POLICY_NAME = 'os_compute_api:wrs-pci'


wrs_pci_policies = [
    policy.DocumentedRuleDefault(
        BASE_POLICY_NAME,
        base.RULE_ADMIN_OR_OWNER,
        """List and show PCI device usage.""",
        [
            {
                'method': 'GET',
                'path': '/wrs-pci'
            },
            {
                'method': 'GET',
                'path': '/wrs-pci/{id}'
            },
        ]),
]


def list_rules():
    return wrs_pci_policies
