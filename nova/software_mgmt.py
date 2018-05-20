#
# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Software Management: SysinvClient, Patching query API.
"""
from keystoneauth1 import loading as keystone
from keystoneauth1 import session
from oslo_log import log as logging

import nova.conf
import requests


CONF = nova.conf.CONF
LOG = logging.getLogger(__name__)

PATCHING_API_PORT = 5487
SYSINV_API_VERSION = 1
AUTHTOKEN_GROUP = 'keystone_authtoken'


class SysinvClient(object):

    def __init__(self):
        self._sysinv = None
        self._auth = None
        self._session = None

    def __enter__(self):
        if not self._connect():
            raise Exception('sysinv failed to get session')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()

    def __del__(self):
        self._disconnect()

    def _disconnect(self):
        """Disconnect from session """
        self._auth = None
        self._session = None

    def _connect(self):
        """Get keystone auth_token and session."""
        if self._session is not None:
            self._disconnect()

        if not self._auth:
            self._auth = keystone.load_auth_from_conf_options(CONF,
                                                              AUTHTOKEN_GROUP)
        if not self._session:
            self._session = session.Session(auth=self._auth)

        return self._session is not None

    @property
    def sysinv(self):
        """Return SysinvClient."""
        # To prevent TOX dependency failures, this import is localized.
        from cgtsclient import client as cgts_client

        self._sysinv = cgts_client.get_client(SYSINV_API_VERSION, **{
            'os_username': self._session.auth._username,
            'os_project_name': self._session.auth._project_name,
            'os_password': self._session.auth._password,
            'os_auth_url': self._session.auth.auth_url + '/v3',
            'os_user_domain_name': self._session.auth._user_domain_name,
            'os_project_domain_name': self._session.auth._project_domain_name,
            'os_endpoint_type': 'internal'})

        return self._sysinv


def patch_query_hosts():
    """Return patch state information for all hosts."""

    # Query WRS patching information for all hosts. Note that patching services
    # endpoints are not usable without patching or admin credentials.
    # This uses the patching internal API that is not generally exposed.
    api_addr = "127.0.0.1:%s" % PATCHING_API_PORT
    url = "http://%s/patch/query_hosts" % api_addr
    timeout = 10
    data = {}
    try:
        r = requests.get(url,
                         headers={'Connection': 'close'},
                         timeout=timeout)
        data = r.json()
        if 'error' in data and data["error"] != "":
            raise ValueError(data["error"])
        if 'data' not in data:
            raise ValueError("No patch data")
    except Exception as ex:
        data['data'] = []
        LOG.exception("Could not get: %(url)s, error=%(err)s",
                      {'url': url, 'err': ex})

    return data
