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
# Copyright (c) 2016-2017 Wind River Systems, Inc.
#

"""The WRS Server Resources Extension."""

from webob import exc

from nova.api.openstack import api_version_request
from nova.api.openstack import common
from nova.api.openstack.compute.schemas import wrs_server_resources as schema
from nova.api.openstack import wsgi
from nova.api import validation
from nova import compute
from nova import exception
from nova.objects import instance_numa_topology as it_obj
from nova.pci import manager as pci_manager
from nova.pci import utils as pci_utils
from nova.policies import wrs_server_resources as wrs_res_policies
from nova import utils
from oslo_log import log as logging


LOG = logging.getLogger(__name__)


class WrsServerResourcesController(wsgi.Controller):
    def __init__(self):
        self.compute_api = compute.API()

    def _get_server(self, context, req, instance_uuid):
        """Utility function for looking up an instance by uuid."""
        instance = common.get_instance(self.compute_api, context,
                                       instance_uuid)
        return instance

    def _get_numa_topology(self, context, instance):
        try:
            numa_topology = it_obj.InstanceNUMATopology.get_by_instance_uuid(
                context, instance['uuid'])
        except exception.NumaTopologyNotFound:
            LOG.warning("Instance does not have numa_topology",
                        instance=instance)
            numa_topology = None

        if not numa_topology:
            # Print mock summary when no topology available - assume 4K pgsize
            return (
                'node:-, %(mem)5dMB, pgsize:%(sz)sK, vcpus:%(vcpus)s'
                % {'mem': instance['memory_mb'],
                   'sz': 4,
                   'vcpus': instance['vcpus']}
            )

        return utils.format_instance_numa_topology(
            numa_topology=numa_topology, instance=instance, delim='\n')

    def _get_pci_devices(self, instance):
        # Get pci_devices associated with instance (i.e., at destination).
        pci_devices = pci_manager.get_instance_pci_devs_by_host_and_node(
            instance, request_id='all')
        if not pci_devices:
            return ''
        return pci_utils.format_instance_pci_devices(
            pci_devices=pci_devices, delim='\n')

    def _extend_server(self, context, server, instance):
        vcpus = instance["vcpus"]
        server["wrs-res:vcpus"] = [instance.get("min_vcpus") or vcpus,
                                   vcpus,
                                   instance.get("max_vcpus") or vcpus
                                  ]
        server["wrs-res:topology"] = self._get_numa_topology(context, instance)
        server["wrs-res:pci_devices"] = self._get_pci_devices(instance)

    def _scale(self, req, instance_id, resource, direction):
        """Begin the scale process with given instance."""
        context = req.environ["nova.context"]
        context.can(wrs_res_policies.BASE_POLICY_NAME)
        instance = self._get_server(context, req, instance_id)

        try:
            self.compute_api.scale(context, instance, resource, direction)
        except (exception.OverQuota,
                exception.CannotScaleBeyondLimits,
                exception.CannotOfflineCpu,
                exception.InstanceScalingError) as scale_error:
            raise exc.HTTPBadRequest(explanation=scale_error.message)
        except exception.InstanceInvalidState as state_error:
            common.raise_http_conflict_for_instance_invalid_state(
                                            state_error, 'scale', instance_id)

    @wsgi.extends
    def show(self, req, resp_obj, id):
        context = req.environ['nova.context']
        if context.can(wrs_res_policies.BASE_POLICY_NAME, fatal=False):
            server = resp_obj.obj['server']
            db_instance = req.get_db_instance(server['id'])
            # server['id'] is guaranteed to be in the cache due to
            # the core API adding it in its 'show' method.
            if api_version_request.wrs_is_supported(req):
                self._extend_server(context, server, db_instance)

    @wsgi.extends
    def detail(self, req, resp_obj):
        context = req.environ['nova.context']
        if context.can(wrs_res_policies.BASE_POLICY_NAME, fatal=False):
            servers = list(resp_obj.obj['servers'])
            for server in servers:
                db_instance = req.get_db_instance(server['id'])
                # server['id'] is guaranteed to be in the cache due to
                # the core API adding it in its 'detail' method.
                if api_version_request.wrs_is_supported(req):
                    self._extend_server(context, server, db_instance)

    @wsgi.response(202)
    @wsgi.action('wrs-res:scale')
    @validation.schema(schema.scale)
    def _action_scale(self, req, id, body):
        """Scales a given instance resource up or down.

        Validation is done via the specified schema.
        """
        direction = body["wrs-res:scale"]["direction"]
        resource = body["wrs-res:scale"]["resource"]
        return self._scale(req, id, resource, direction)
