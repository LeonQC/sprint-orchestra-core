plugins {
    java
    id("org.springframework.boot")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    runtimeOnly("io.micrometer:micrometer-registry-prometheus")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    // Explicit for IntelliJ IDEA runner classpath (transitive dep is sometimes omitted from the IDE model).
    implementation("org.hibernate.common:hibernate-commons-annotations:6.0.6.Final")
    implementation("org.springframework.boot:spring-boot-starter-data-redis")
    implementation("org.springframework.boot:spring-boot-starter-cache")
    // GitHub login keeps its state in the HTTP session between the redirect and the callback; in
    // Redis, the callback works on whichever instance it lands on.
    implementation("org.springframework.session:spring-session-data-redis")
    implementation("org.springframework.boot:spring-boot-starter-security")
    implementation("org.springframework.boot:spring-boot-starter-oauth2-client")
    implementation("org.springframework.boot:spring-boot-starter-oauth2-resource-server")
    implementation("org.springframework.kafka:spring-kafka")
    implementation("org.springframework.security:spring-security-oauth2-jose")
    implementation("org.springdoc:springdoc-openapi-starter-webmvc-ui:2.5.0")
    implementation("org.flywaydb:flyway-core")
    implementation(platform("software.amazon.awssdk:bom:2.31.72"))
    implementation("software.amazon.awssdk:s3")

    implementation("org.postgresql:postgresql:42.7.3")

    // OpenSearch owns all text search in this service: the tiered global search in SearchService.
    //
    // OpenSearch rather than Elasticsearch because the deployment target is Amazon OpenSearch
    // Service. The query DSL is untouched by that choice (OpenSearch forked from Elasticsearch 7.10
    // and everything SearchQueries builds predates the fork), but the CLIENT had to change: since
    // 7.14 the Elasticsearch client verifies an `X-Elastic-Product` response header and refuses to
    // talk to anything that is not Elasticsearch. The OpenSearch client is a fork of the Elastic
    // one, so the builder API below is identical apart from package names.
    //
    // The low-level java client rather than a Spring Data starter: the tiered search issues four
    // hand-built queries per request (prefix, simple_query_string, wildcard, fuzzy) and combines
    // them itself, so a repository/entity-mapping abstraction would be something to work around
    // rather than something to use.
    //
    // Version pinned here directly: the Spring Boot BOM manages `co.elastic.clients` but knows nothing about `org.opensearch.client`,
    // so an explicit version is honoured instead of being silently overridden. Keep it in lockstep
    // with the opensearch image tag in docker-compose.yml and vectorization-service's Python pin.
    implementation("org.opensearch.client:opensearch-java:2.25.0")
    implementation("org.opensearch.client:opensearch-rest-client:2.19.0")

    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")

    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test> {
    useJUnitPlatform()
}
