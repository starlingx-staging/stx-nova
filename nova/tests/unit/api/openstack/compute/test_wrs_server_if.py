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
# Copyright (c) 2015-2017 Wind River Systems, Inc.
#

from oslo_serialization import jsonutils

from nova.api.openstack import wsgi as os_wsgi
from nova import objects
from nova import test
from nova.tests.unit.api.openstack import fakes

UUID1 = '00000000-0000-0000-0000-000000000001'
UUID2 = '00000000-0000-0000-0000-000000000002'
UUID3 = '00000000-0000-0000-0000-000000000003'
NW_CACHE = [
    {
        'address': 'aa:aa:aa:aa:aa:aa',
        'id': 1,
        'vif_model': 'virtio',
        'network': {
            'bridge': 'br0',
            'id': 1,
            'label': 'private',
            'subnets': [
                {
                    'cidr': '192.168.1.0/24',
                    'ips': [
                        {
                            'address': '192.168.1.100',
                            'type': 'fixed',
                            'floating_ips': [
                                {'address': '5.0.0.1', 'type': 'floating'},
                            ],
                        },
                    ],
                },
            ]
        }
    },
    {
        'address': 'bb:bb:bb:bb:bb:bb',
        'id': 2,
        'vif_model': None,
        'network': {
            'bridge': 'br1',
            'id': 2,
            'label': 'public',
            'subnets': [
                {
                    'cidr': '10.0.0.0/24',
                    'ips': [
                        {
                            'address': '10.0.0.100',
                            'type': 'fixed',
                            'floating_ips': [
                                {'address': '5.0.0.2', 'type': 'floating'},
                            ],
                        }
                    ],
                },
            ]
        }
    }
]
ALL_NICS = []
for index, cache in enumerate(NW_CACHE):
    name = 'nic' + str(index + 1)
    nic = {name: {'port_id': cache['id'],
                  'mac_address': cache['address'],
                  'vif_model': cache['vif_model'],
                  'vif_pci_address': '',
                  'mtu': None,  # only available from neutron in real env
                  'network': cache['network']['label']}}
    ALL_NICS.append(nic)


def fake_compute_get(*args, **kwargs):
    inst = fakes.stub_instance_obj(None, 1, uuid=UUID3, nw_cache=NW_CACHE)
    return inst


def fake_compute_get_all(*args, **kwargs):
    inst_list = [
        fakes.stub_instance_obj(None, 1, uuid=UUID1, nw_cache=NW_CACHE),
        fakes.stub_instance_obj(None, 2, uuid=UUID2, nw_cache=NW_CACHE),
    ]
    return objects.InstanceList(objects=inst_list)


class WrsServerIfTestV21(test.TestCase):
    content_type = 'application/json'
    prefix = 'wrs-if'
    _prefix = "/v2/fake"
    wsgi_api_version = os_wsgi.DEFAULT_API_VERSION

    def setUp(self):
        super(WrsServerIfTestV21, self).setUp()
        self.flags(use_neutron=False)
        fakes.stub_out_nw_api(self)
        self.stub_out('nova.compute.api.API.get', fake_compute_get)
        self.stub_out('nova.compute.api.API.get_all', fake_compute_get_all)
        return_server = fakes.fake_instance_get()
        self.stub_out('nova.db.instance_get_by_uuid', return_server)

    def _make_request(self, url):
        req = fakes.HTTPRequest.blank(url)
        req.accept = self.content_type
        res = req.get_response(self._get_app())
        return res

    def _get_app(self):
        return fakes.wsgi_app_v21()

    def _get_server(self, body):
        return jsonutils.loads(body).get('server')

    def _get_servers(self, body):
        return jsonutils.loads(body).get('servers')

    def _get_nics(self, server):
        return server['wrs-if:nics']

    def assertServerNics(self, server):
        self.assertEqual(ALL_NICS, self._get_nics(server))

    def test_show(self):
        url = self._prefix + '/servers/%s' % UUID3
        res = self._make_request(url)
        self.assertEqual(res.status_int, 200)
        self.assertServerNics(self._get_server(res.body))

    def test_detail(self):
        url = self._prefix + '/servers/detail'
        res = self._make_request(url)
        self.assertEqual(res.status_int, 200)
        for i, server in enumerate(self._get_servers(res.body)):
            self.assertServerNics(server)
