package com.jiraagentic.app.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3ClientBuilder;
import software.amazon.awssdk.services.s3.S3Configuration;

import java.net.URI;

@Configuration
public class S3StorageConfig {

    /**
     * Defaults point at local MinIO. On AWS, set the endpoint and keys to empty: the client then uses
     * real S3 and the ECS task role, so no access key is stored anywhere.
     */
    @Bean
    public S3Client s3Client(
            @Value("${app.attachments.s3.endpoint:http://localhost:9000}") String endpoint,
            @Value("${app.attachments.s3.region:us-east-1}") String region,
            @Value("${app.attachments.s3.access-key:minioadmin}") String accessKey,
            @Value("${app.attachments.s3.secret-key:minioadmin}") String secretKey,
            @Value("${app.attachments.s3.path-style-access:true}") boolean pathStyleAccess) {

        S3ClientBuilder builder = S3Client.builder()
                .region(Region.of(region))
                .serviceConfiguration(S3Configuration.builder().pathStyleAccessEnabled(pathStyleAccess).build());
        if (!endpoint.isBlank()) {
            builder.endpointOverride(URI.create(endpoint));
        }
        builder.credentialsProvider(accessKey.isBlank()
                ? DefaultCredentialsProvider.builder().build()
                : StaticCredentialsProvider.create(AwsBasicCredentials.create(accessKey, secretKey)));
        return builder.build();
    }
}
