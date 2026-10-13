import os
from dataclasses import dataclass

import requests

from openpilot.common.api import API_HOST, Api
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "openpilot-navigation"
HTTP_TIMEOUT = 10  # seconds
NEAR_BOX_DEG = 0.5  # half size of the box results are biased to, about 50 km
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"


@dataclass
class Place:
  name: str
  details: str
  latitude: float
  longitude: float
  distance: float | None = None  # route length in meters
  duration: float | None = None  # driving time in seconds

  def to_param(self) -> dict:
    return {"latitude": self.latitude, "longitude": self.longitude, "place_name": self.name, "place_details": self.details}


def parse_place(result: dict) -> Place:
  address = result.get("address", {})
  street = " ".join(s for s in (address.get("house_number"), address.get("road")) if s)
  # a named place (shop, park) leads with its name, a plain address with house number and street
  name = result.get("name") or street or result["display_name"].split(",")[0]
  city = address.get("city") or address.get("town") or address.get("village") or address.get("county") or ""
  area = address.get("suburb") or address.get("neighbourhood") or ""
  details = ", ".join(s for s in (street if street != name else "", area, city) if s)
  return Place(name, details, float(result["lat"]), float(result["lon"]))


def _request(params: dict) -> list[Place]:
  resp = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
  resp.raise_for_status()
  return [parse_place(r) for r in resp.json()]


def search(query: str, near: tuple[float, float] | None = None, limit: int = 5) -> list[Place]:
  params: dict[str, str | int] = {"q": query, "format": "jsonv2", "addressdetails": 1, "limit": limit}
  if near is not None:
    # places around you first, nominatim only ranks by importance otherwise ("main st" lands in another country)
    lat, lon = near
    viewbox = f"{lon - NEAR_BOX_DEG},{lat + NEAR_BOX_DEG},{lon + NEAR_BOX_DEG},{lat - NEAR_BOX_DEG}"
    places = _request({**params, "viewbox": viewbox, "bounded": 1})
    if places:
      return places
  return _request(params)


def _directions(coords: str) -> tuple[float, float]:
  # the same directions api navd routes with: mapbox with a token, else comma's proxy
  token = os.environ.get("MAPBOX_TOKEN")
  host = "https://api.mapbox.com" if token else API_HOST.rstrip('/') + '/v1/nav'
  if not token:
    token = Api(Params().get("DongleId")).get_token()
  resp = requests.get(f"{host}/directions/v5/mapbox/driving-traffic/{coords}",
                      params={'access_token': token, 'overview': 'false', 'steps': 'false'}, timeout=HTTP_TIMEOUT)
  resp.raise_for_status()
  route = resp.json()['routes'][0]
  return route['distance'], route['duration']


def _osrm(coords: str) -> tuple[float, float]:
  # public OSM router, for devices the proxy doesn't serve. no traffic in its times
  resp = requests.get(OSRM_URL + coords, params={'overview': 'false'}, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
  resp.raise_for_status()
  route = resp.json()['routes'][0]
  return route['distance'], route['duration']


def add_route_summary(place: Place, start: tuple[float, float]) -> None:
  """Fill in the route length and driving time from start (latitude, longitude) to the place, if a router answers."""
  coords = f"{start[1]},{start[0]};{place.longitude},{place.latitude}"
  for name, router in (("directions", _directions), ("osrm", _osrm)):
    try:
      place.distance, place.duration = router(coords)
      return
    except Exception:
      cloudlog.exception(f"route summary from {name} failed")
