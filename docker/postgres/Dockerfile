# Keep the existing PostgreSQL 15.17 / Debian 13 base (including libc/collations). Add only
# pgvector's PG15 runtime files. Its official build uses portable OPTFLAGS and the library links
# only libc; the older Debian 12 runtime ABI works on this newer base. Both inputs are pinned.
ARG VECTOR_IMAGE=pgvector/pgvector:pg15@sha256:18d16372b8406bb38a9f94cbff15d125c463d71fde2770aa8b5c64bfcc1578ee
ARG POSTGRES_IMAGE=postgres:15@sha256:0f989d656f95dda60a631c8e1c67287df49b31a73f147e4290d5924fdc7f46e3
FROM ${VECTOR_IMAGE} AS vector
FROM ${POSTGRES_IMAGE}
COPY --from=vector /usr/lib/postgresql/15/lib/vector.so /usr/lib/postgresql/15/lib/vector.so
COPY --from=vector /usr/share/postgresql/15/extension/vector* /usr/share/postgresql/15/extension/
COPY --from=vector /usr/share/doc/pgvector /usr/share/doc/pgvector
