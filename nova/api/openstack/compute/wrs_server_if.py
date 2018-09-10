#   Licensed under the Apache License, Version 2.0 (the "License"); you may
#   not use this file except in compliance with the License. You may obtain
#   a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   License for the specific language governing permissions and limitations
#   under the License.
#
# Copyright (c) 2014-2017 Wind River Systems, Inc.
#

"""The WRS Server Groups Extension."""

from nova.api.openstack import api_version_request
from nova.api.openstack import common
from nova.api.openstack import wsgi
from nova.policies import wrs_server_if as wrs_if_policies


# Note that this limit is arbitrary, and exists as an engineering limit
# on the maximum supported number of NICs configured for a VM.
MAXIMUM_VNICS = 16


class WrsServerIfController(wsgi.Controller):
    def __init__(self, *args, **kwargs):
        super(WrsServerIfController, self).__init__(*args, **kwargs)

    def _extend_server(self, context, server, instance):
        server["wrs-if:nics"] = common.get_nics_for_instance(context, instance)

    @wsgi.extends
    def show(self, req, resp_obj, id):
        context = req.environ['nova.context']
        if context.can(wrs_if_policies.BASE_POLICY_NAME, fatal=False):
            server = resp_obj.obj['server']
            db_instance = req.get_db_instance(server['id'])
            # server['id'] is guaranteed to be in the cache due to
            # the core API adding it in its 'show' method.
            if api_version_request.wrs_is_supported(req):
                self._extend_server(context, server, db_instance)

    @wsgi.extends
    def detail(self, req, resp_obj):
        context = req.environ['nova.context']
        if context.can(wrs_if_policies.BASE_POLICY_NAME, fatal=False):
            servers = list(resp_obj.obj['servers'])
            for server in servers:
                instance = req.get_db_instance(server['id'])
                # server['id'] is guaranteed to be in the cache due to
                # the core API adding it in its 'detail' method.
                if api_version_request.wrs_is_supported(req):
                    self._extend_server(context, server, instance)
