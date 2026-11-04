"""Google Cloud names used by the acquisition and release scripts.

The thesis used one private Google Cloud project. Set these environment
variables to your own project and buckets before running a cloud script.
"""
import os

GCP_PROJECT = os.environ.get('GCP_PROJECT', 'my-gcp-project')
# Bucket of the expanded maritime RGB release (Mother, D1-D3).
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'my-maritime-rgb-bucket')
# Bucket of the earlier RGB super-resolution pilot (Sentinel-2 / SkySat).
GCS_PILOT_BUCKET = os.environ.get('GCS_PILOT_BUCKET', 'my-rgb-sr-pilot-bucket')
