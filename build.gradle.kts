plugins {
    id("org.springframework.boot") apply false
}

import org.springframework.boot.gradle.tasks.run.BootRun

allprojects {
    group = "com.jiraagentic"
    version = "0.0.1-SNAPSHOT"
}

subprojects {
    repositories {
        mavenCentral()
    }
}

val springBootBomVersion = "3.2.0"

subprojects {
    plugins.withId("org.springframework.boot") {
        dependencies {
            val bootBom = enforcedPlatform("org.springframework.boot:spring-boot-dependencies:$springBootBomVersion")
            add("implementation", bootBom)
            add("compileOnly", bootBom)
            add("annotationProcessor", bootBom)
            add("testImplementation", bootBom)
            add("testRuntimeOnly", bootBom)
        }
    }
}

// Forward environment variables or Gradle project properties into bootRun.
fun resolveBootRunEnv(key: String): String? =
    providers.environmentVariable(key).orNull
        ?: providers.gradleProperty(key).orNull

val bootRunEnvKeys = listOf(
    "GITHUB_OAUTH_ENABLED",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "GITHUB_OAUTH_REDIRECT_URI",
    "GITHUB_TOKEN",
    "APP_GITHUB_TOKEN",
    "APP_KAFKA_ATTACHMENT_INGESTION_ENABLED",
    "APP_KAFKA_ATTACHMENT_INGESTION_TOPIC",
    "APP_JWT_PRIVATE_KEY",
    "APP_JWT_PRIVATE_KEY_PATH",
    "APP_JWT_ALLOW_GENERATED_KEY",
    "APP_JWT_JWK_SET_URI",
    "INTERNAL_SERVICE_TOKEN",
    "SPRING_KAFKA_BOOTSTRAP_SERVERS",
)

subprojects {
    plugins.withId("org.springframework.boot") {
        tasks.withType<BootRun>().configureEach {
            bootRunEnvKeys.forEach { key ->
                resolveBootRunEnv(key)?.let { value -> environment(key, value) }
            }
        }
    }
}

fun Exec.killPort(port: Int) {
    group = "dev"
    commandLine(
        "bash",
        "-c",
        "pid=\$(lsof -ti tcp:$port 2>/dev/null || true); if [ -n \"\$pid\" ]; then kill \$pid 2>/dev/null || true; fi",
    )
}

tasks.register<Exec>("devStopBackend") {
    description = "Stop any process listening on backend port 8081"
    killPort(8081)
}

tasks.register("devStopDevPorts") {
    group = "dev"
    description = "Stop backend (8081) before compound run configs"
    dependsOn("devStopBackend")
}
