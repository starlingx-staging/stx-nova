#
# Copyright (c) 2015 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

"""The WRS PCI Device Extension."""

from nova.api.openstack import extensions
from nova.api.openstack import wsgi
from nova.conf import CONF
from nova import objects
from nova.pci import devspec
from nova.pci import whitelist
from nova.policies import wrs_pci as wrs_pci_policies

from oslo_log import log as logging


LOG = logging.getLogger(__name__)


class WrsPciUsage(object):
    def __init__(self, product_id, vendor_id, class_id):
        '''Construct a WrsPciUsage object with the given values.'''
        self.product_id = product_id
        self.vendor_id = vendor_id
        self.class_id = class_id
        self.pci_pfs_configured = 0
        self.pci_pfs_count = 0
        self.pci_vfs_configured = 0
        self.pci_vfs_count = 0

    def get_device_usage_by_node(self, node):
        if node.pci_device_pools and node.pci_device_pools.objects:
            for pool in node.pci_device_pools.objects:
                if (self.product_id != devspec.ANY and
                    self.product_id != pool.product_id):
                    continue
                if (self.class_id and
                    self.class_id != pool.tags.get("class_id")):
                    continue
                tags = pool.tags
                dev_type = tags.get('dev_type', {})
                configured = int(tags.get('configured', '0'))

                if ((self.product_id == pool.product_id and
                     self.vendor_id == pool.vendor_id) or
                     self.class_id == pool.tags.get("class_id")):
                    if 'type-PCI' in dev_type:
                        self.pci_pfs_configured += configured
                        self.pci_pfs_count += pool.count

                    if 'type-VF' in dev_type:
                        self.pci_vfs_configured += configured
                        self.pci_vfs_count += pool.count

                    if not self.class_id:
                        # Class id may not have been specified in the spec.
                        # Fill it out here, which makes the replies / display
                        # more consistent.
                        self.class_id = pool.tags.get("class_id")


class WrsPciController(wsgi.Controller):
    def __init__(self):
        super(WrsPciController, self).__init__()

    @extensions.expected_errors(404)
    def index(self, req):

        """Returns list of devices.

        The list is filtered by device id or alias if specified in the request,
        else all devices are returned.
        """

        context = req.environ['nova.context']
        context.can(wrs_pci_policies.BASE_POLICY_NAME)

        nodes = objects.ComputeNodeList.get_all(context)

        # The pci alias parameter has the same format as the PCI whitelist.
        # Load the list of devices that are enabled for passthrough to a
        # guest, and show resource usage for each.
        pci_filter = whitelist.Whitelist(CONF.pci.alias)

        device_usage = []
        for spec in pci_filter.specs:
            tags = spec.get_tags()
            dev_name = tags.get('name')
            device_id = spec.product_id

            # Optional request to list by (PCI alias) name, or device id.
            if 'device' in req.GET:
                req_device = req.GET['device']
                if req_device == dev_name or req_device == spec.product_id:
                    device_id = spec.product_id
                else:
                    continue

            class_id = tags.get("class_id", None)
            usage = WrsPciUsage(device_id, spec.vendor_id, class_id)
            for node in nodes:
                usage.get_device_usage_by_node(node)

            if (usage.pci_pfs_configured > 0 or
                usage.pci_vfs_configured > 0):
                device_usage.append({'device_name': str(dev_name),
                    'device_id': usage.product_id,
                    'vendor_id': usage.vendor_id,
                    'class_id': usage.class_id,
                    'pci_pfs_configured': usage.pci_pfs_configured,
                    'pci_pfs_used':
                        usage.pci_pfs_configured - usage.pci_pfs_count,
                    'pci_vfs_configured': usage.pci_vfs_configured,
                    'pci_vfs_used':
                        usage.pci_vfs_configured - usage.pci_vfs_count})
        return {'pci_device_usage': device_usage}

    @extensions.expected_errors(404)
    def show(self, req, id):

        """Returns list of host usage for a particular device.

        The list is filtered by host if specified in the request,
        else all hosts are returned.

        Note that the id parameter can be the PCI alias or PCI
        device id.
        """

        context = req.environ['nova.context']
        context.can(wrs_pci_policies.BASE_POLICY_NAME)
        host = None

        if 'host' in req.GET:
            host = req.GET['host']
        if host:
            nodes = objects.ComputeNodeList.get_all_by_host(
                           context, host)
        else:
            nodes = objects.ComputeNodeList.get_all(context)

        # The pci alias parameter has the same format as the PCI whitelist.
        # Load the list of devices that are enabled for passthrough to a
        # guest, and show resource usage for each.
        pci_filter = whitelist.Whitelist(CONF.pci.alias)

        device_usage = []
        for spec in pci_filter.specs:
            tags = spec.get_tags()

            # Devices can be shown by (PCI alias) name, or device id.
            dev_name = tags.get('name')
            if id == dev_name:
                device_id = spec.product_id
            else:
                device_id = id

            if device_id != spec.product_id:
                continue

            for node in nodes:
                class_id = tags.get("class_id", None)
                usage = WrsPciUsage(device_id, spec.vendor_id, class_id)
                usage.get_device_usage_by_node(node)
                device_usage.append({'device_name': str(dev_name),
                    'device_id': usage.product_id,
                    'vendor_id': usage.vendor_id,
                    'class_id': usage.class_id,
                    'host': node.host,
                    'pci_pfs_configured': usage.pci_pfs_configured,
                    'pci_pfs_used':
                        usage.pci_pfs_configured - usage.pci_pfs_count,
                    'pci_vfs_configured': usage.pci_vfs_configured,
                    'pci_vfs_used':
                        usage.pci_vfs_configured - usage.pci_vfs_count})
        return {'pci_device_usage': device_usage}
