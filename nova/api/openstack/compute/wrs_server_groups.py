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
from nova.api.openstack import wsgi
from nova import exception
from nova import objects
from nova.policies import wrs_server_groups as wrs_sg_policies


class WrsServerGroupController(wsgi.Controller):
    def _get_server_group(self, context, instance):
        try:
            sg = objects.InstanceGroup.get_by_instance_uuid(context,
                                                            instance.uuid)
        except exception.InstanceGroupNotFound:
            sg = None
        if not sg:
            return ''

        return "%s (%s)" % (sg.name, sg.uuid)

    @wsgi.extends
    def show(self, req, resp_obj, id):
        context = req.environ['nova.context']
        if context.can(wrs_sg_policies.BASE_POLICY_NAME, fatal=False):
            server = resp_obj.obj['server']
            instance = req.get_db_instance(server['id'])
            # server['id'] is guaranteed to be in the cache due to
            # the core API adding it in its 'show' method.
            if api_version_request.wrs_is_supported(req):
                server["wrs-sg:server_group"] = self._get_server_group(
                                                                context,
                                                                instance)

    @wsgi.extends
    def detail(self, req, resp_obj):
        context = req.environ['nova.context']
        if context.can(wrs_sg_policies.BASE_POLICY_NAME, fatal=False):
            servers = list(resp_obj.obj['servers'])
            for server in servers:
                instance = req.get_db_instance(server['id'])
                # server['id'] is guaranteed to be in the cache due to
                # the core API adding it in its 'detail' method.
                if api_version_request.wrs_is_supported(req):
                    server["wrs-sg:server_group"] = self._get_server_group(
                                                                    context,
                                                                    instance)
