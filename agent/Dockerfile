# Agent image: a static binary on distroless (~10 MB).
# shortcut: no journalctl inside, so journald sources only work with the binary installed on the host; ship a debian-based variant if container users need journald.
FROM --platform=$BUILDPLATFORM golang:1.23-alpine AS build
ARG TARGETOS TARGETARCH TARGETVARIANT VERSION=dev
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH GOARM=${TARGETVARIANT#v} go build -trimpath -ldflags "-s -w -X main.version=$VERSION" -o /logpipe-agent .

FROM gcr.io/distroless/static-debian12
COPY --from=build /logpipe-agent /usr/local/bin/logpipe-agent
COPY docker/agent.yaml /etc/logpipe-agent/agent.yaml
VOLUME /var/lib/logpipe-agent
ENTRYPOINT ["/usr/local/bin/logpipe-agent"]
CMD ["run"]
