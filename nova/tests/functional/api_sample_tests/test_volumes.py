# Copyright 2012 Nebula, Inc.
# Copyright 2014 IBM Corp.
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

import datetime

from oslo_utils.fixture import uuidsentinel as uuids

from nova import context
from nova import objects
from nova.tests import fixtures
from nova.tests.functional.api_sample_tests import api_sample_base
from nova.tests.functional.api_sample_tests import test_servers
from nova.tests.unit.api.openstack import fakes
from nova.tests.unit import fake_block_device
from nova.tests.unit import fake_instance


class SnapshotsSampleJsonTests(api_sample_base.ApiSampleTestBaseV21):
    sample_dir = "os-volumes"

    create_subs = {
            'snapshot_name': 'snap-001',
            'description': 'Daily backup',
            'volume_id': '521752a6-acf6-4b2d-bc7a-119f9148cd8c'
    }

    def setUp(self):
        super(SnapshotsSampleJsonTests, self).setUp()
        self.stub_out("nova.volume.cinder.API.get_all_snapshots",
                      fakes.stub_snapshot_get_all)
        self.stub_out("nova.volume.cinder.API.get_snapshot",
                      fakes.stub_snapshot_get)

    def _create_snapshot(self):
        self.stub_out("nova.volume.cinder.API.create_snapshot",
                      fakes.stub_snapshot_create)

        response = self._do_post("os-snapshots",
                                 "snapshot-create-req",
                                 self.create_subs)
        return response

    def test_snapshots_create(self):
        response = self._create_snapshot()
        self._verify_response("snapshot-create-resp",
                              self.create_subs, response, 200)

    def test_snapshots_delete(self):
        self.stub_out("nova.volume.cinder.API.delete_snapshot",
                      fakes.stub_snapshot_delete)
        self._create_snapshot()
        response = self._do_delete('os-snapshots/100')
        self.assertEqual(202, response.status_code)
        self.assertEqual('', response.text)

    def test_snapshots_detail(self):
        response = self._do_get('os-snapshots/detail')
        self._verify_response('snapshots-detail-resp', {}, response, 200)

    def test_snapshots_list(self):
        response = self._do_get('os-snapshots')
        self._verify_response('snapshots-list-resp', {}, response, 200)

    def test_snapshots_show(self):
        response = self._do_get('os-snapshots/100')
        subs = {
            'snapshot_name': 'Default name',
            'description': 'Default description'
        }
        self._verify_response('snapshots-show-resp', subs, response, 200)


def _get_volume_id():
    return 'a26887c6-c47b-4654-abb5-dfadf7d3f803'


def _stub_volume(id, displayname="Volume Name",
                 displaydesc="Volume Description", size=100):
    volume = {
              'id': id,
              'size': size,
              'availability_zone': 'zone1:host1',
              'status': 'in-use',
              'attach_status': 'attached',
              'name': 'vol name',
              'display_name': displayname,
              'display_description': displaydesc,
              'created_at': datetime.datetime(2008, 12, 1, 11, 1, 55),
              'snapshot_id': None,
              'volume_type_id': 'fakevoltype',
              'volume_metadata': [],
              'volume_type': {'name': 'Backup'},
              'multiattach': False,
              'attachments': {'3912f2b4-c5ba-4aec-9165-872876fe202e':
                              {'mountpoint': '/',
                               'attachment_id':
                                   'a26887c6-c47b-4654-abb5-dfadf7d3f803'
                               }
                              }
              }
    return volume


def _stub_volume_get(stub_self, context, volume_id):
    return _stub_volume(volume_id)


def _stub_volume_delete(stub_self, context, *args, **param):
    pass


def _stub_volume_get_all(stub_self, context, search_opts=None):
    id = _get_volume_id()
    return [_stub_volume(id)]


def _stub_volume_create(stub_self, context, size, name, description,
                        snapshot, **param):
    id = _get_volume_id()
    return _stub_volume(id)


class VolumesSampleJsonTest(test_servers.ServersSampleBase):
    sample_dir = "os-volumes"

    def setUp(self):
        super(VolumesSampleJsonTest, self).setUp()
        fakes.stub_out_networking(self)

        self.stub_out("nova.volume.cinder.API.delete",
                      _stub_volume_delete)
        self.stub_out("nova.volume.cinder.API.get", _stub_volume_get)
        self.stub_out("nova.volume.cinder.API.get_all",
                      _stub_volume_get_all)

    def _post_volume(self):
        subs_req = {
                'volume_name': "Volume Name",
                'volume_desc': "Volume Description",
        }

        self.stub_out("nova.volume.cinder.API.create",
                      _stub_volume_create)
        response = self._do_post('os-volumes', 'os-volumes-post-req',
                                 subs_req)
        self._verify_response('os-volumes-post-resp', subs_req, response, 200)

    def test_volumes_show(self):
        subs = {
                'volume_name': "Volume Name",
                'volume_desc': "Volume Description",
        }
        vol_id = _get_volume_id()
        response = self._do_get('os-volumes/%s' % vol_id)
        self._verify_response('os-volumes-get-resp', subs, response, 200)

    def test_volumes_index(self):
        subs = {
                'volume_name': "Volume Name",
                'volume_desc': "Volume Description",
        }
        response = self._do_get('os-volumes')
        self._verify_response('os-volumes-index-resp', subs, response, 200)

    def test_volumes_detail(self):
        # For now, index and detail are the same.
        # See the volumes api
        subs = {
                'volume_name': "Volume Name",
                'volume_desc': "Volume Description",
        }
        response = self._do_get('os-volumes/detail')
        self._verify_response('os-volumes-detail-resp', subs, response, 200)

    def test_volumes_create(self):
        self._post_volume()

    def test_volumes_delete(self):
        self._post_volume()
        vol_id = _get_volume_id()
        response = self._do_delete('os-volumes/%s' % vol_id)
        self.assertEqual(202, response.status_code)
        self.assertEqual('', response.text)


class VolumeAttachmentsSample(test_servers.ServersSampleBase):
    sample_dir = "os-volumes"

    OLD_VOLUME_ID = 'a26887c6-c47b-4654-abb5-dfadf7d3f803'
    NEW_VOLUME_ID = 'a26887c6-c47b-4654-abb5-dfadf7d3f805'

    def _get_tags_per_volume(self):
        """Allows subclasses to override which volumes have tags

        :returns: dict, keyed by volume ID, to tag value; if a volume ID is
            not found in the resulting dict it is assumed to not have a tag
        """
        return {}

    def _stub_db_bdms_get_all_by_instance(self, server_id):

        def fake_bdms_get_all_by_instance(context, instance_uuid,
                                          use_subordinate=False):
            bdms = [
                fake_block_device.FakeDbBlockDeviceDict(
                {'id': 1, 'volume_id': self.OLD_VOLUME_ID,
                'instance_uuid': server_id, 'source_type': 'volume',
                'destination_type': 'volume', 'device_name': '/dev/sdd'}),
                fake_block_device.FakeDbBlockDeviceDict(
                {'id': 2, 'volume_id': 'a26887c6-c47b-4654-abb5-dfadf7d3f804',
                'instance_uuid': server_id, 'source_type': 'volume',
                'destination_type': 'volume', 'device_name': '/dev/sdc'})
            ]
            tags_per_volume = self._get_tags_per_volume()
            for bdm_dict in bdms:
                bdm_dict['tag'] = tags_per_volume.get(bdm_dict['volume_id'])
            return bdms

        self.stub_out('nova.db.api.block_device_mapping_get_all_by_instance',
                      fake_bdms_get_all_by_instance)

    def fake_bdm_get_by_volume_and_instance(
            self, ctxt, volume_id, instance_uuid, expected_attrs=None):
        tag = self._get_tags_per_volume().get(self.OLD_VOLUME_ID)
        return objects.BlockDeviceMapping._from_db_object(
            ctxt, objects.BlockDeviceMapping(),
            fake_block_device.FakeDbBlockDeviceDict(
                {'id': 1, 'volume_id': self.OLD_VOLUME_ID,
                 'instance_uuid': instance_uuid, 'source_type': 'volume',
                 'destination_type': 'volume', 'device_name': '/dev/sdd',
                 'tag': tag})
        )

    def _stub_compute_api_get(self):

        def fake_compute_api_get(self, context, instance_id,
                                 expected_attrs=None,
                                 cell_down_support=False):
            return fake_instance.fake_instance_obj(
                    context, **{'uuid': instance_id})

        self.stub_out('nova.compute.api.API.get', fake_compute_api_get)

    def _get_vol_attachment_subs(self, subs):
        """Allows subclasses to override/supplement request/response subs"""
        return subs

    def test_attach_volume_to_server(self):
        self.stub_out('nova.volume.cinder.API.get', fakes.stub_volume_get)
        self.stub_out('nova.volume.cinder.API.attachment_create',
                      lambda *a, **k: {'id': uuids.volume})
        device_name = '/dev/vdd'
        bdm = objects.BlockDeviceMapping()
        bdm['device_name'] = device_name
        self.stub_out(
            'nova.compute.manager.ComputeManager.reserve_block_device_name',
            lambda *a, **k: bdm)
        self.stub_out(
            'nova.compute.manager.ComputeManager.attach_volume',
            lambda *a, **k: None)

        volume = fakes.stub_volume_get(None, context.get_admin_context(),
                                       'a26887c6-c47b-4654-abb5-dfadf7d3f803')
        subs = {
            'volume_id': volume['id'],
            'device': device_name
        }
        server_id = self._post_server()
        subs = self._get_vol_attachment_subs(subs)
        response = self._do_post('servers/%s/os-volume_attachments'
                                 % server_id,
                                 'attach-volume-to-server-req', subs)

        self._verify_response('attach-volume-to-server-resp', subs,
                              response, 200)

    def test_list_volume_attachments(self):
        server_id = self._post_server()
        self._stub_db_bdms_get_all_by_instance(server_id)

        response = self._do_get('servers/%s/os-volume_attachments'
                                % server_id)
        subs = self._get_vol_attachment_subs({})
        self._verify_response('list-volume-attachments-resp', subs,
                              response, 200)

    def test_volume_attachment_detail(self):
        server_id = self._post_server()
        self.stub_out(
            'nova.objects.BlockDeviceMapping.get_by_volume_and_instance',
            self.fake_bdm_get_by_volume_and_instance)
        self._stub_compute_api_get()
        response = self._do_get('servers/%s/os-volume_attachments/%s'
                                % (server_id, self.OLD_VOLUME_ID))
        subs = self._get_vol_attachment_subs({})
        self._verify_response('volume-attachment-detail-resp', subs,
                              response, 200)

    def test_volume_attachment_delete(self):
        server_id = self._post_server()
        self.stub_out(
            'nova.objects.BlockDeviceMapping.get_by_volume_and_instance',
            self.fake_bdm_get_by_volume_and_instance)
        self._stub_compute_api_get()
        self.stub_out('nova.volume.cinder.API.get', fakes.stub_volume_get)
        self.stub_out('nova.compute.api.API.detach_volume',
                      lambda *a, **k: None)
        response = self._do_delete('servers/%s/os-volume_attachments/%s'
                                   % (server_id, self.OLD_VOLUME_ID))
        self.assertEqual(202, response.status_code)
        self.assertEqual('', response.text)

    def test_volume_attachment_update(self):
        self.stub_out('nova.volume.cinder.API.get', fakes.stub_volume_get)
        subs = {
            'volume_id': self.NEW_VOLUME_ID
        }
        server_id = self._post_server()
        self.stub_out(
            'nova.objects.BlockDeviceMapping.get_by_volume_and_instance',
            self.fake_bdm_get_by_volume_and_instance)
        self._stub_compute_api_get()
        self.stub_out('nova.volume.cinder.API.get', fakes.stub_volume_get)
        self.stub_out('nova.compute.api.API.swap_volume',
                      lambda *a, **k: None)
        response = self._do_put('servers/%s/os-volume_attachments/%s'
                                % (server_id, self.OLD_VOLUME_ID),
                                'update-volume-req',
                                subs)
        self.assertEqual(202, response.status_code)
        self.assertEqual('', response.text)


class VolumeAttachmentsSampleV249(VolumeAttachmentsSample):
    sample_dir = "os-volumes"
    microversion = '2.49'
    scenarios = [('v2_49', {'api_major_version': 'v2.1'})]

    def setUp(self):
        super(VolumeAttachmentsSampleV249, self).setUp()
        self.useFixture(fixtures.CinderFixture(self))

    def _get_vol_attachment_subs(self, subs):
        return dict(subs, tag='foo')


class VolumeAttachmentsSampleV270(VolumeAttachmentsSampleV249):
    """2.70 adds the "tag" parameter to the response body"""
    microversion = '2.70'
    scenarios = [('v2_70', {'api_major_version': 'v2.1'})]

    def _get_tags_per_volume(self):
        return {
            self.OLD_VOLUME_ID: 'foo',
            self.NEW_VOLUME_ID: None
        }
