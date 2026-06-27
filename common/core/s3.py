from typing import Any

from aiobotocore.session import get_session

from common.core.config import config


class AsyncS3Client:
    def __init__(self) -> None:
        self.region_name = config.AWS_REGION
        self.session = get_session()
        self.client: Any | None = None

    async def _get_client(self) -> Any:
        if self.client is None:
            self.client = await self.session.create_client(
                "s3",
                region_name=self.region_name,
                aws_access_key_id=config.AWS_ACCESS_KEY,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            ).__aenter__()
        return self.client

    async def put_object(self, Bucket, Key, Body, **kwargs):
        client = await self._get_client()
        return await client.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=Body,
            **kwargs,
        )

    async def generate_presigned_url(self, ClientMethod, Params, ExpiresIn=3600, HttpMethod="GET"):
        client = await self._get_client()
        return await client.generate_presigned_url(
            ClientMethod=ClientMethod,
            Params=Params,
            ExpiresIn=ExpiresIn,
            HttpMethod=HttpMethod,
        )

    async def get_object(self, Bucket, Key):
        client = await self._get_client()
        response = await client.get_object(Bucket=Bucket, Key=Key)

        if not response:
            raise Exception("Unable to download the object from S3")
        # Read the content
        body = await response["Body"].read()

        return body

    async def put_bucket_lifecycle_configuration(self, Bucket, LifecycleConfiguration):
        client = await self._get_client()
        return await client.put_bucket_lifecycle_configuration(
            Bucket=Bucket, LifecycleConfiguration=LifecycleConfiguration
        )

    async def delete_object(self, Bucket, Key):
        client = await self._get_client()
        return await client.delete_object(Bucket=Bucket, Key=Key)

    async def close(self):
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def get_s3_url(self, Bucket, Key, expires_in=2 * 60 * 60):
        return await self.generate_presigned_url(
            "get_object",
            {"Bucket": Bucket, "Key": Key},
            expires_in,
        )


async_s3_client = AsyncS3Client()
