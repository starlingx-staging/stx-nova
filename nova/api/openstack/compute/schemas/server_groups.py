# Copyright 2014 NEC Corporation.  All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
# Copyright (c) 2016-2017 Wind River Systems, Inc.
#
import copy

from nova.api.validation import parameter_types

# NOTE(russellb) There is one other policy, 'legacy', but we don't allow that
# being set via the API.  It's only used when a group gets automatically
# created to support the legacy behavior of the 'group' scheduler hint.
create = {
    'type': 'object',
    'properties': {
        'server_group': {
            'type': 'object',
            'properties': {
                'name': parameter_types.name,
                'policies': {
                    # This allows only a single item and it must be one of the
                    # enumerated values. So this is really just a single string
                    # value, but for legacy reasons is an array. We could
                    # probably change the type from array to string with a
                    # microversion at some point but it's very low priority.
                    'type': 'array',
                    'items': [{
                        'type': 'string',
                        'enum': ['anti-affinity', 'affinity']}],
                    'uniqueItems': True,
                    'additionalItems': False,
                }
            },
            'required': ['name', 'policies'],
            'additionalProperties': False,
        }
    },
    'required': ['server_group'],
    'additionalProperties': False,
}

create_v215 = copy.deepcopy(create)
policies = create_v215['properties']['server_group']['properties']['policies']
policies['items'][0]['enum'].extend(['soft-anti-affinity', 'soft-affinity'])


# WRS:extension, group metadata
positive_integer_not_empty = {
    'type': ['integer', 'string'],
    'pattern': '^[0-9]+$', 'minimum': 1
}

boolean_with_nullstring = copy.deepcopy(parameter_types.boolean)
boolean_with_nullstring['type'].append('null')
boolean_with_nullstring['enum'].extend(['', None])


server_group_metadata_create = {
    'type': 'object',
    'properties': {
        'wrs-sg:group_size': positive_integer_not_empty,
        'wrs-sg:best_effort': parameter_types.boolean,
    },
    'additionalProperties': False,
}

server_group_metadata = {
    'type': 'object',
    'properties': {
        'wrs-sg:group_size': {
            'oneOf': [
                positive_integer_not_empty,
                {
                    'type': ['null', 'string'],
                    'enum': [None, ''],
                },
            ]
        },
        'wrs-sg:best_effort': boolean_with_nullstring,
    },
    'additionalProperties': False,
}

# extend group creation to include optional metadata
properties = create_v215['properties']['server_group']['properties']
properties['metadata'] = server_group_metadata_create

properties['project_id'] = parameter_types.project_id

# schema for setting metadata on existing group
set_meta = {
    'type': 'object',
    'properties': {
        'set_metadata': {
            'type': 'object',
            'properties': {
                'metadata': server_group_metadata
            },
            'required': ['metadata'],
            'additionalProperties': False,
        }
    },
    'required': ['set_metadata'],
    'additionalProperties': False,
}
