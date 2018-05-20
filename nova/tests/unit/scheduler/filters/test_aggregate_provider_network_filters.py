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
# Copyright (c) 2014-2017 Wind River Systems, Inc.
#

import collections
import mock


from nova import context
from nova import objects
from nova import test

from nova.scheduler.filters import aggregate_provider_network_filter

from nova.tests.unit.scheduler import fakes


@mock.patch('nova.scheduler.filters.utils.aggregate_metadata_get_by_host')
class TestAggregateNetworkProviderFilter(test.NoDBTestCase):

    def setUp(self):
        super(TestAggregateNetworkProviderFilter, self).setUp()
        self.filt_cls = \
            aggregate_provider_network_filter.AggregateProviderNetworkFilter()
        self.ctxt = context.get_admin_context()

    def test_agg_network_provider_filter_fails_if_blank(self, agg_mock):
        agg_mock.return_value = collections.defaultdict(set)
        host = fakes.FakeHostState('host1', 'node1', {})

        spec_obj = objects.RequestSpec(self.ctxt)
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints=None)
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_agg_network_provider_filter_fails_if_mismatched(self, agg_mock):
        agg_mock.return_value = {'provider:physical_network': ['physnet22', ]}
        host = fakes.FakeHostState('host1', 'node1', {})

        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

        # Almost matches (too short)
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet2', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

        # Almost matches (too long)
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet222', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_agg_network_provider_filter_passes(self, agg_mock):
        agg_mock.return_value = {'provider:physical_network': ['physnet0', ]}
        host = fakes.FakeHostState('host1', 'node1', {})

        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

        agg_mock.return_value = {'provider:physical_network':
                                     ['physnet0_again', ]}
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0_again', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_agg_network_provider_filter_multi(self, agg_mock):
        # 2 diff physical networks. Test we can match either.
        agg_mock.return_value = {'provider:physical_network':
                                     ['physnet0', 'physnet1', ]}
        host = fakes.FakeHostState('host1', 'node1', {})

        # Match first
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet0', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

        # Match second
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                           {'provider:physical_network': ['physnet1', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

        # Match both
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                   {'provider:physical_network': ['physnet0', 'physnet1', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

        # Match both with duplicate
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                   {'provider:physical_network': ['physnet1', 'physnet0',
                                                  'physnet1', ]})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

        # Returns false if scheduler wants both, but host only provides one.
        agg_mock.return_value = {'provider:physical_network': ['physnet1', ]}
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                   {'provider:physical_network': ['physnet0', 'physnet1', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))
        spec_obj = objects.RequestSpec(self.ctxt, scheduler_hints =
                   {'provider:physical_network': ['physnet1', 'physnet0', ]})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))
