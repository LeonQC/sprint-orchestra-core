# Build from the repo root (Gradle multi-project):
#   docker build -f jira-backend/Dockerfile -t sprint-orchestra/jira-backend .
FROM eclipse-temurin:17-jdk AS build
WORKDIR /src
COPY gradlew settings.gradle.kts build.gradle.kts ./
COPY gradle ./gradle
COPY jira-backend/build.gradle.kts jira-backend/
# Dependencies first, so source-only changes reuse this layer.
RUN ./gradlew :jira-backend:dependencies --no-daemon -q > /dev/null
COPY jira-backend/src jira-backend/src
RUN ./gradlew :jira-backend:bootJar --no-daemon -q && cp jira-backend/build/libs/jira-backend-*-SNAPSHOT.jar /app.jar

FROM eclipse-temurin:17-jre
RUN useradd --system --create-home app
USER app
WORKDIR /home/app
COPY --from=build /app.jar app.jar
EXPOSE 8081
ENV JAVA_OPTS="-XX:MaxRAMPercentage=75"
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]
