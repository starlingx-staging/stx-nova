# Copyright (c) 2012 Citrix Systems, Inc.
# Copyright 2010 OpenStack Foundation
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

"""
Management class for Pool-related functions (join, eject, etc).
"""

from oslo_log import log as logging
from oslo_serialization import jsonutils
import six
import six.moves.urllib.parse as urlparse

from nova.compute import rpcapi as compute_rpcapi
import nova.conf
from nova import exception
from nova.i18n import _
from nova.virt.xenapi import pool_states
from nova.virt.xenapi import vm_utils

LOG = logging.getLogger(__name__)

CONF = nova.conf.CONF


class ResourcePool(object):
    """Implements resource pool operations."""
    def __init__(self, session, virtapi):
        host_rec = session.host.get_record(session.host_ref)
        self._host_name = host_rec['hostname']
        self._host_addr = host_rec['address']
        self._host_uuid = host_rec['uuid']
        self._session = session
        self._virtapi = virtapi
        self.compute_rpcapi = compute_rpcapi.ComputeAPI()

    def undo_aggregate_operation(self, context, op, aggregate,
                                  host, set_error):
        """Undo aggregate operation when pool error raised."""
        try:
            if set_error:
                metadata = {pool_states.KEY: pool_states.ERROR}
                aggregate.update_metadata(metadata)
            op(host)
        except Exception:
            LOG.exception(_('Aggregate %(aggregate_id)s: unrecoverable '
                            'state during operation on %(host)s'),
                          {'aggregate_id': aggregate.id, 'host': host})

    def add_to_aggregate(self, context, aggregate, host, subordinate_info=None):
        """Add a compute host to an aggregate."""
        if not pool_states.is_hv_pool(aggregate.metadata):
            return

        if CONF.xenserver.independent_compute:
            raise exception.NotSupportedWithOption(
                operation='adding to a XenServer pool',
                option='CONF.xenserver.independent_compute')

        invalid = {pool_states.CHANGING: _('setup in progress'),
                   pool_states.DISMISSED: _('aggregate deleted'),
                   pool_states.ERROR: _('aggregate in error')}

        if (aggregate.metadata[pool_states.KEY] in invalid.keys()):
            raise exception.InvalidAggregateActionAdd(
                    aggregate_id=aggregate.id,
                    reason=invalid[aggregate.metadata[pool_states.KEY]])

        if (aggregate.metadata[pool_states.KEY] == pool_states.CREATED):
            aggregate.update_metadata({pool_states.KEY: pool_states.CHANGING})
        if len(aggregate.hosts) == 1:
            # this is the first host of the pool -> make it main
            self._init_pool(aggregate.id, aggregate.name)
            # save metadata so that we can find the main again
            metadata = {'main_compute': host,
                        host: self._host_uuid,
                        pool_states.KEY: pool_states.ACTIVE}
            aggregate.update_metadata(metadata)
        else:
            # the pool is already up and running, we need to figure out
            # whether we can serve the request from this host or not.
            main_compute = aggregate.metadata['main_compute']
            if main_compute == CONF.host and main_compute != host:
                # this is the main ->  do a pool-join
                # To this aim, nova compute on the subordinate has to go down.
                # NOTE: it is assumed that ONLY nova compute is running now
                self._join_subordinate(aggregate.id, host,
                                 subordinate_info.get('compute_uuid'),
                                 subordinate_info.get('url'), subordinate_info.get('user'),
                                 subordinate_info.get('passwd'))
                metadata = {host: subordinate_info.get('xenhost_uuid'), }
                aggregate.update_metadata(metadata)
            elif main_compute and main_compute != host:
                # send rpc cast to main, asking to add the following
                # host with specified credentials.
                subordinate_info = self._create_subordinate_info()

                self.compute_rpcapi.add_aggregate_host(
                    context, host, aggregate, main_compute, subordinate_info)

    def remove_from_aggregate(self, context, aggregate, host, subordinate_info=None):
        """Remove a compute host from an aggregate."""
        subordinate_info = subordinate_info or dict()
        if not pool_states.is_hv_pool(aggregate.metadata):
            return

        invalid = {pool_states.CREATED: _('no hosts to remove'),
                   pool_states.CHANGING: _('setup in progress'),
                   pool_states.DISMISSED: _('aggregate deleted')}
        if aggregate.metadata[pool_states.KEY] in invalid.keys():
            raise exception.InvalidAggregateActionDelete(
                    aggregate_id=aggregate.id,
                    reason=invalid[aggregate.metadata[pool_states.KEY]])

        main_compute = aggregate.metadata['main_compute']
        if main_compute == CONF.host and main_compute != host:
            # this is the main -> instruct it to eject a host from the pool
            host_uuid = aggregate.metadata[host]
            self._eject_subordinate(aggregate.id,
                              subordinate_info.get('compute_uuid'), host_uuid)
            aggregate.update_metadata({host: None})
        elif main_compute == host:
            # Remove main from its own pool -> destroy pool only if the
            # main is on its own, otherwise raise fault. Destroying a
            # pool made only by main is fictional
            if len(aggregate.hosts) > 1:
                # NOTE: this could be avoided by doing a main
                # re-election, but this is simpler for now.
                raise exception.InvalidAggregateActionDelete(
                                    aggregate_id=aggregate.id,
                                    reason=_('Unable to eject %s '
                                             'from the pool; pool not empty')
                                             % host)
            self._clear_pool(aggregate.id)
            aggregate.update_metadata({'main_compute': None, host: None})
        elif main_compute and main_compute != host:
            # A main exists -> forward pool-eject request to main
            subordinate_info = self._create_subordinate_info()

            self.compute_rpcapi.remove_aggregate_host(
                context, host, aggregate.id, main_compute, subordinate_info)
        else:
            # this shouldn't have happened
            raise exception.AggregateError(aggregate_id=aggregate.id,
                                           action='remove_from_aggregate',
                                           reason=_('Unable to eject %s '
                                           'from the pool; No main found')
                                           % host)

    def _join_subordinate(self, aggregate_id, host, compute_uuid, url, user, passwd):
        """Joins a subordinate into a XenServer resource pool."""
        try:
            args = {'compute_uuid': compute_uuid,
                    'url': url,
                    'user': user,
                    'password': passwd,
                    'force': jsonutils.dumps(CONF.xenserver.use_join_force),
                    'main_addr': self._host_addr,
                    'main_user': CONF.xenserver.connection_username,
                    'main_pass': CONF.xenserver.connection_password, }
            self._session.call_plugin('xenhost.py', 'host_join', args)
        except self._session.XenAPI.Failure as e:
            LOG.error("Pool-Join failed: %s", e)
            raise exception.AggregateError(aggregate_id=aggregate_id,
                                           action='add_to_aggregate',
                                           reason=_('Unable to join %s '
                                                  'in the pool') % host)

    def _eject_subordinate(self, aggregate_id, compute_uuid, host_uuid):
        """Eject a subordinate from a XenServer resource pool."""
        try:
            # shutdown nova-compute; if there are other VMs running, e.g.
            # guest instances, the eject will fail. That's a precaution
            # to deal with the fact that the admin should evacuate the host
            # first. The eject wipes out the host completely.
            vm_ref = self._session.VM.get_by_uuid(compute_uuid)
            self._session.VM.clean_shutdown(vm_ref)

            host_ref = self._session.host.get_by_uuid(host_uuid)
            self._session.pool.eject(host_ref)
        except self._session.XenAPI.Failure as e:
            LOG.error("Pool-eject failed: %s", e)
            raise exception.AggregateError(aggregate_id=aggregate_id,
                                           action='remove_from_aggregate',
                                           reason=six.text_type(e.details))

    def _init_pool(self, aggregate_id, aggregate_name):
        """Set the name label of a XenServer pool."""
        try:
            pool_ref = self._session.pool.get_all()[0]
            self._session.pool.set_name_label(pool_ref, aggregate_name)
        except self._session.XenAPI.Failure as e:
            LOG.error("Unable to set up pool: %s.", e)
            raise exception.AggregateError(aggregate_id=aggregate_id,
                                           action='add_to_aggregate',
                                           reason=six.text_type(e.details))

    def _clear_pool(self, aggregate_id):
        """Clear the name label of a XenServer pool."""
        try:
            pool_ref = self._session.pool.get_all()[0]
            self._session.pool.set_name_label(pool_ref, '')
        except self._session.XenAPI.Failure as e:
            LOG.error("Pool-set_name_label failed: %s", e)
            raise exception.AggregateError(aggregate_id=aggregate_id,
                                           action='remove_from_aggregate',
                                           reason=six.text_type(e.details))

    def _create_subordinate_info(self):
        """XenServer specific info needed to join the hypervisor pool."""
        # replace the address from the xenapi connection url
        # because this might be 169.254.0.1, i.e. xenapi
        # NOTE: password in clear is not great, but it'll do for now
        sender_url = swap_xapi_host(
            CONF.xenserver.connection_url, self._host_addr)

        return {
            "url": sender_url,
            "user": CONF.xenserver.connection_username,
            "passwd": CONF.xenserver.connection_password,
            "compute_uuid": vm_utils.get_this_vm_uuid(None),
            "xenhost_uuid": self._host_uuid,
        }


def swap_xapi_host(url, host_addr):
    """Replace the XenServer address present in 'url' with 'host_addr'."""
    temp_url = urlparse.urlparse(url)
    return url.replace(temp_url.hostname, '%s' % host_addr)
