#
# Copyright (c) 2015 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

"""The WRS Provider Network Extension."""

from neutronclient.common import exceptions as n_exc
from nova.api.openstack import extensions
from nova.api.openstack import wsgi
from nova.i18n import _
from nova.network.neutronv2 import api as neutronapi
from nova import objects
from nova.policies import wrs_providernets as wrs_providernet_policies

from oslo_log import log as logging
from webob import exc


LOG = logging.getLogger(__name__)


ALIAS = "wrs-providernet"


class WrsController(wsgi.Controller):
    def __init__(self):
        super(WrsController, self).__init__()

    @extensions.expected_errors(404)
    def show(self, req, id):
        context = req.environ['nova.context']
        context.can(wrs_providernet_policies.BASE_POLICY_NAME)

        neutron = neutronapi.get_client(context)
        try:
            providernet = neutron.show_providernet(id).get('providernet', {})
        except n_exc.NeutronClientException:
            LOG.exception("Neutron Error getting provider network %s", id)
            msg = _("Error getting provider network")
            raise exc.HTTPNotFound(explanation=msg)

        physnet = providernet.get('name')
        pci_pfs_configured = 0
        pci_pfs_count = 0
        pci_vfs_configured = 0
        pci_vfs_count = 0
        nodes = objects.ComputeNodeList.get_all(context)
        for node in nodes:
            if node.pci_device_pools and node.pci_device_pools.objects:
                for pool in node.pci_device_pools.objects:
                    tags = pool.tags
                    if physnet in tags.get('physical_network', {}):
                        dev_type = tags.get('dev_type', {})
                        configured = int(tags.get('configured', '0'))

                        if 'type-PF' in dev_type:
                            pci_pfs_configured += configured
                            pci_pfs_count += pool.count

                        if 'type-VF' in dev_type:
                            pci_vfs_configured += configured
                            pci_vfs_count += pool.count

        return {'providernet':
                {'id': id, 'name': physnet,
                 'pci_pfs_configured': pci_pfs_configured,
                 'pci_pfs_used': pci_pfs_configured - pci_pfs_count,
                 'pci_vfs_configured': pci_vfs_configured,
                 'pci_vfs_used': pci_vfs_configured - pci_vfs_count}}
