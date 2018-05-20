# Copyright (c) 2016-2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#


scale = {
    'type': 'object',
    'properties': {
        'wrs-res:scale': {
            'type': 'object',
            'properties': {
                'resource': {
                    'type': 'string',
                    'enum': ['cpu']
                },
                'direction': {
                    'type': 'string',
                    'enum': ['up', 'down']
                }
            },
            'required': ['resource', 'direction'],
            'additionalProperties': False
        }
    },
    'required': ['wrs-res:scale'],
    'additionalProperties': False
}
