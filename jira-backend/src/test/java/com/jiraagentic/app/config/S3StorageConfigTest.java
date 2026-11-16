package com.jiraagentic.app.config;

import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.services.s3.S3Client;

import java.net.URI;

import static org.assertj.core.api.Assertions.assertThat;

class S3StorageConfigTest {

    private final S3StorageConfig config = new S3StorageConfig();

    @Test
    void localDefaultsTalkToMinioWithItsKeys() {
        try (S3Client client = config.s3Client("http://localhost:9000", "us-east-1", "minioadmin", "minioadmin", true)) {
            assertThat(client.serviceClientConfiguration().endpointOverride()).contains(URI.create("http://localhost:9000"));
            assertThat(client.serviceClientConfiguration().credentialsProvider()).isInstanceOf(StaticCredentialsProvider.class);
        }
    }

    @Test
    void blankEndpointAndKeysUseRealS3AndTheTaskRole() {
        try (S3Client client = config.s3Client("", "us-east-1", "", "", false)) {
            assertThat(client.serviceClientConfiguration().endpointOverride()).isEmpty();
            assertThat(client.serviceClientConfiguration().credentialsProvider()).isInstanceOf(DefaultCredentialsProvider.class);
        }
    }
}
