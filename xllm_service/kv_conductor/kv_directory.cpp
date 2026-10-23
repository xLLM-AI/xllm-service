/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kv_conductor/kv_directory.h"

#include <glog/logging.h>
#include <zmq.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "kv_conductor/kv_event_decoder.h"

namespace xllm_service {
namespace {

using Socket = std::unique_ptr<void, decltype(&zmq_close)>;
using Clock = std::chrono::steady_clock;

bool bounded_integer(const nlohmann::json& object,
                     const std::string& key,
                     uint64_t minimum,
                     uint64_t maximum,
                     uint64_t* output) {
  if (!object.contains(key)) {
    return *output >= minimum && *output <= maximum;
  }
  const auto& value = object[key];
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<int64_t>() < 0)) {
    return false;
  }
  const uint64_t number = value.get<uint64_t>();
  if (number < minimum || number > maximum) {
    return false;
  }
  *output = number;
  return true;
}

bool connectable_endpoint(const std::string& address) {
  if (address.compare(0, 6, "tcp://") != 0) {
    return false;
  }
  const size_t delimiter = address.rfind(':');
  if (delimiter <= 6 || delimiter + 1 >= address.size()) {
    return false;
  }
  const std::string host = address.substr(6, delimiter - 6);
  if (host == "*" || host == "0.0.0.0" || host == "[::]") {
    return false;
  }
  uint32_t port = 0;
  for (size_t index = delimiter + 1; index < address.size(); ++index) {
    const char digit = address[index];
    if (digit < '0' || digit > '9' || port > 6553) {
      return false;
    }
    port = port * 10 + static_cast<uint32_t>(digit - '0');
  }
  return port > 0 && port <= 65535;
}

Socket make_socket(void* context, int32_t type, size_t max_message_bytes) {
  Socket socket(zmq_socket(context, type), &zmq_close);
  if (!socket) {
    return socket;
  }
  const int32_t linger = 0;
  const int32_t hwm = 128;
  const int64_t maximum = static_cast<int64_t>(max_message_bytes);
  if (zmq_setsockopt(socket.get(), ZMQ_LINGER, &linger, sizeof(linger)) != 0 ||
      zmq_setsockopt(socket.get(), ZMQ_RCVHWM, &hwm, sizeof(hwm)) != 0 ||
      zmq_setsockopt(socket.get(), ZMQ_SNDHWM, &hwm, sizeof(hwm)) != 0 ||
      zmq_setsockopt(socket.get(), ZMQ_MAXMSGSIZE, &maximum, sizeof(maximum)) !=
          0) {
    socket.reset();
  }
  return socket;
}

// Called only after poll readiness. Every frame is consumed without blocking;
// malformed/oversized messages make the caller recreate the socket session.
bool receive_frames(void* socket,
                    size_t maximum,
                    std::vector<std::string>* frames) {
  frames->clear();
  size_t total = 0;
  bool more = true;
  while (more && frames->size() < 4) {
    zmq_msg_t message;
    if (zmq_msg_init(&message) != 0) {
      return false;
    }
    const int32_t size = zmq_msg_recv(&message, socket, ZMQ_DONTWAIT);
    if (size < 0) {
      zmq_msg_close(&message);
      return false;
    }
    total += zmq_msg_size(&message);
    more = zmq_msg_more(&message) != 0;
    if (total > maximum) {
      zmq_msg_close(&message);
      return false;
    }
    frames->emplace_back(static_cast<const char*>(zmq_msg_data(&message)),
                         zmq_msg_size(&message));
    zmq_msg_close(&message);
  }
  return !more;
}

bool is_snapshot(const KVEventBatch& batch) {
  return !batch.events.empty() &&
         batch.events.front().type == KVEventType::CLEARED;
}

bool same_source(const KVEventSource& left, const KVEventSource& right) {
  return left.instance_name == right.instance_name &&
         left.incarnation_id == right.incarnation_id &&
         (!left.incarnation_id.empty() ||
          left.register_ts_ms == right.register_ts_ms) &&
         left.dp_rank == right.dp_rank && left.endpoint == right.endpoint &&
         left.replay_endpoint == right.replay_endpoint &&
         left.topic == right.topic && left.block_size == right.block_size &&
         left.hash_seed == right.hash_seed;
}

}  // namespace

bool load_kv_directory_config(const std::string& path,
                              KVDirectoryConfig* config,
                              std::string* error) {
  const auto fail = [&](const std::string& message) {
    *error = message;
    return false;
  };
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 0 || input.tellg() > 1024 * 1024) {
    return fail("KV events config must be a readable file of at most 1 MiB");
  }
  const size_t size = static_cast<size_t>(input.tellg());
  std::string text(size, '\0');
  input.seekg(0);
  if (!input.read(text.data(), static_cast<std::streamsize>(size))) {
    return fail("failed to read KV events config");
  }
  const auto json = nlohmann::json::parse(text, nullptr, false);
  if (!json.is_object()) {
    return fail("KV events config must be an object");
  }
  KVDirectoryConfig result;
  uint64_t max_streams = result.max_streams;
  uint64_t blocks = result.max_blocks_per_stream;
  uint64_t message_bytes = result.max_message_bytes;
  uint64_t buffered_bytes = result.max_buffered_bytes;
  uint64_t timeout = result.replay_timeout_ms;
  uint64_t interval = result.resync_interval_ms;
  if (!bounded_integer(json, "max_streams", 1, 128, &max_streams) ||
      !bounded_integer(json, "max_blocks_per_stream", 1, 10000000, &blocks) ||
      !bounded_integer(
          json, "max_message_bytes", 1024, 256 * 1024 * 1024, &message_bytes) ||
      !bounded_integer(json,
                       "max_buffered_bytes",
                       message_bytes,
                       512 * 1024 * 1024,
                       &buffered_bytes) ||
      !bounded_integer(json, "replay_timeout_ms", 50, 60000, &timeout) ||
      !bounded_integer(
          json, "resync_interval_ms", timeout, 3600000, &interval)) {
    return fail("invalid KV events resource limits or timeouts");
  }
  result.max_streams = static_cast<size_t>(max_streams);
  result.max_blocks_per_stream = static_cast<size_t>(blocks);
  result.max_message_bytes = static_cast<size_t>(message_bytes);
  result.max_buffered_bytes = static_cast<size_t>(buffered_bytes);
  result.replay_timeout_ms = static_cast<int32_t>(timeout);
  result.resync_interval_ms = static_cast<int32_t>(interval);
  if (json.contains("discovery")) {
    const auto& discovery = json["discovery"];
    if (!discovery.is_object() ||
        (discovery.contains("enabled") && !discovery["enabled"].is_boolean())) {
      return fail("discovery must be an object with a boolean enabled field");
    }
    result.discovery.enabled = discovery.value("enabled", true);
    if (discovery.contains("topic") && !discovery["topic"].is_string()) {
      return fail("discovery topic must be a string");
    }
    result.discovery.topic = discovery.value("topic", std::string());
    uint64_t block_size = result.discovery.block_size;
    uint64_t seed = result.discovery.hash_seed;
    uint64_t event_offset = result.discovery.event_port_offset;
    uint64_t replay_offset = result.discovery.replay_port_offset;
    uint64_t refresh = result.discovery.refresh_interval_ms;
    if (!bounded_integer(discovery, "block_size", 1, 8192, &block_size) ||
        !bounded_integer(discovery,
                         "hash_seed",
                         0,
                         std::numeric_limits<uint64_t>::max(),
                         &seed) ||
        !bounded_integer(
            discovery, "event_port_offset", 0, 65534, &event_offset) ||
        !bounded_integer(
            discovery, "replay_port_offset", 0, 65534, &replay_offset) ||
        !bounded_integer(
            discovery, "refresh_interval_ms", 50, 60000, &refresh) ||
        event_offset == replay_offset) {
      return fail("invalid KV discovery port offsets or limits");
    }
    result.discovery.block_size = static_cast<uint32_t>(block_size);
    result.discovery.hash_seed = seed;
    result.discovery.event_port_offset = static_cast<uint32_t>(event_offset);
    result.discovery.replay_port_offset = static_cast<uint32_t>(replay_offset);
    result.discovery.refresh_interval_ms = static_cast<int32_t>(refresh);
  }
  const auto streams = json.value("streams", nlohmann::json::array());
  if (!streams.is_array() || streams.size() > result.max_streams ||
      (result.discovery.enabled && !streams.empty()) ||
      (!result.discovery.enabled && streams.empty())) {
    return fail(
        "configure either discovery or a nonempty bounded streams list");
  }
  std::set<std::pair<std::string, int32_t>> identities;
  std::set<std::string> endpoints;
  result.sources.reserve(streams.size());
  for (const auto& stream : streams) {
    if (!stream.is_object()) {
      return fail("each KV stream must be an object");
    }
    for (const std::string& key :
         {"instance_name", "endpoint", "replay_endpoint"}) {
      if (!stream.contains(key) || !stream[key].is_string() ||
          stream[key].get_ref<const std::string&>().empty()) {
        return fail("each KV stream requires a nonempty " + key);
      }
    }
    if (stream.contains("topic") && !stream["topic"].is_string()) {
      return fail("KV stream topic must be a string");
    }
    KVEventSource source;
    source.instance_name = stream["instance_name"].get<std::string>();
    source.endpoint = stream["endpoint"].get<std::string>();
    source.replay_endpoint = stream["replay_endpoint"].get<std::string>();
    source.topic = stream.value("topic", std::string());
    uint64_t rank = 0;
    uint64_t block_size = source.block_size;
    uint64_t seed = source.hash_seed;
    if (!bounded_integer(stream, "dp_rank", 0, 65535, &rank) ||
        !bounded_integer(stream, "block_size", 1, 8192, &block_size) ||
        !bounded_integer(stream,
                         "hash_seed",
                         0,
                         std::numeric_limits<uint64_t>::max(),
                         &seed)) {
      return fail("invalid KV stream rank, block size, or hash seed");
    }
    source.dp_rank = static_cast<int32_t>(rank);
    source.block_size = static_cast<uint32_t>(block_size);
    source.hash_seed = seed;
    if (!connectable_endpoint(source.endpoint) ||
        !connectable_endpoint(source.replay_endpoint) ||
        !identities.emplace(source.instance_name, source.dp_rank).second ||
        !endpoints.insert(source.endpoint).second ||
        !endpoints.insert(source.replay_endpoint).second) {
      return fail(
          "KV streams require unique identities and connectable TCP endpoints");
    }
    result.sources.emplace_back(std::move(source));
  }
  *config = std::move(result);
  error->clear();
  return true;
}

class KVDirectory::Stream final {
 public:
  Stream(KVEventSource source, const KVDirectoryConfig& config)
      : source_(std::move(source)),
        index_(source_.block_size,
               source_.hash_seed,
               config.max_blocks_per_stream),
        config_(config) {}

  ~Stream() { stop(); }

  void start() {
    thread_ = std::thread([this] { run(); });
  }

  void stop() {
    request_stop();
    if (thread_.joinable()) {
      thread_.join();
    }
    index_.invalidate("subscriber stopped");
  }

  void request_stop() { stopped_.store(true); }

  KVPrefixMatch match(const std::vector<int32_t>& tokens) const {
    KVPrefixMatch result = index_.match(tokens);
    if (stopped_.load()) {
      result = KVPrefixMatch{};
      result.error = "subscriber stopped";
    }
    return result;
  }

  KVEventSource source_;
  KVIndex index_;

 private:
  void run() {
    while (!stopped_.load()) {
      run_session();
      index_.invalidate("event transport disconnected");
      for (int32_t retry = 0; retry < 10 && !stopped_.load(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
  }

  void run_session() {
    std::unique_ptr<void, decltype(&zmq_ctx_term)> context(zmq_ctx_new(),
                                                           &zmq_ctx_term);
    if (!context) {
      return;
    }
    Socket subscriber =
        make_socket(context.get(), ZMQ_SUB, config_.max_message_bytes);
    Socket monitor =
        make_socket(context.get(), ZMQ_PAIR, config_.max_message_bytes);
    Socket replay(nullptr, &zmq_close);
    const auto reset_replay = [&]() {
      replay =
          make_socket(context.get(), ZMQ_DEALER, config_.max_message_bytes);
      return replay &&
             zmq_connect(replay.get(), source_.replay_endpoint.c_str()) == 0;
    };
    if (!subscriber || !monitor ||
        zmq_setsockopt(subscriber.get(),
                       ZMQ_SUBSCRIBE,
                       source_.topic.data(),
                       source_.topic.size()) != 0 ||
        zmq_socket_monitor(subscriber.get(),
                           "inproc://kv-monitor",
                           ZMQ_EVENT_CONNECTED | ZMQ_EVENT_DISCONNECTED) != 0 ||
        zmq_connect(monitor.get(), "inproc://kv-monitor") != 0 ||
        zmq_connect(subscriber.get(), source_.endpoint.c_str()) != 0 ||
        !reset_replay()) {
      LOG(ERROR) << "Cannot connect KV event stream " << source_.instance_name
                 << ": " << zmq_strerror(zmq_errno());
      return;
    }
    std::vector<KVEventBatch> pending;
    size_t pending_bytes = 0;
    std::optional<KVEventBatch> snapshot;
    bool in_flight = false;
    // Pre-request increments are covered by the upcoming snapshot and must not
    // postpone its retry deadline while the index is unsynchronized.
    bool awaiting_snapshot = true;
    bool subscription_connected = false;
    auto deadline = Clock::now();
    auto next_request = Clock::now();
    const auto recover = [&](const std::string& reason) {
      LOG(WARNING) << "Resynchronizing KV stream " << source_.instance_name
                   << "/" << source_.dp_rank << ": " << reason;
      index_.invalidate(reason);
      pending.clear();
      pending_bytes = 0;
      snapshot.reset();
      in_flight = false;
      awaiting_snapshot = true;
      next_request = Clock::now() + std::chrono::milliseconds(200);
      return reset_replay();
    };
    std::vector<std::string> frames;
    while (!stopped_.load()) {
      const auto now = Clock::now();
      if (in_flight && now >= deadline && !recover("snapshot timeout")) {
        return;
      }
      if (subscription_connected && !in_flight && now >= next_request) {
        // xLLM ROUTER receives [identity, empty delimiter, uint64 start_seq].
        // It replies with a full current snapshot followed by UINT64_MAX.
        const std::array<uint8_t, 8> start{};
        if (zmq_send(replay.get(), nullptr, 0, ZMQ_DONTWAIT | ZMQ_SNDMORE) <
                0 ||
            zmq_send(replay.get(), start.data(), start.size(), ZMQ_DONTWAIT) <
                0) {
          if (!recover("cannot request snapshot")) {
            return;
          }
          continue;
        }
        in_flight = true;
        deadline = now + std::chrono::milliseconds(config_.replay_timeout_ms);
      }
      zmq_pollitem_t items[] = {{monitor.get(), 0, ZMQ_POLLIN, 0},
                                {subscriber.get(), 0, ZMQ_POLLIN, 0},
                                {replay.get(), 0, ZMQ_POLLIN, 0}};
      const int32_t polled = zmq_poll(items, 3, 50);
      if (polled < 0) {
        if (zmq_errno() == EINTR) {
          continue;
        }
        return;
      }
      if (items[0].revents & ZMQ_POLLIN) {
        if (!receive_frames(monitor.get(), 1024, &frames) ||
            frames.size() != 2 || frames[0].size() < sizeof(uint16_t)) {
          return;
        }
        uint16_t event = 0;
        std::memcpy(&event, frames[0].data(), sizeof(event));
        if (event == ZMQ_EVENT_DISCONNECTED) {
          return;
        }
        if (event == ZMQ_EVENT_CONNECTED &&
            !recover("subscription connection changed")) {
          return;
        }
        subscription_connected = true;
        // A replaced replay socket invalidates this poll's readiness bits.
        continue;
      }
      if (items[1].revents & ZMQ_POLLIN) {
        if (!receive_frames(
                subscriber.get(), config_.max_message_bytes, &frames)) {
          return;
        }
        KVEventBatch batch;
        if (frames.size() != 3 || frames[0] != source_.topic ||
            !decode_kv_batch(frames[1],
                             frames[2],
                             source_.dp_rank,
                             source_.block_size,
                             &batch)) {
          if (!recover("invalid live event batch")) {
            return;
          }
          continue;
        }
        if (is_snapshot(batch)) {
          if (!index_.apply(batch)) {
            if (!recover("invalid live snapshot")) {
              return;
            }
            continue;
          }
          pending.clear();
          pending_bytes = 0;
          snapshot.reset();
          in_flight = false;
          awaiting_snapshot = false;
          next_request = Clock::now() +
                         std::chrono::milliseconds(config_.resync_interval_ms);
          if (!reset_replay()) {
            return;
          }
          continue;
        }
        if (in_flight) {
          // Bound decoded-record overhead as well as MessagePack bytes.
          pending_bytes += frames[2].size() + sizeof(KVEventBatch) +
                           batch.events.capacity() * sizeof(KVEvent);
          for (const auto& event : batch.events) {
            pending_bytes += event.hashes.capacity() * sizeof(uint64_t) +
                             event.token_ids.capacity() * sizeof(int32_t);
          }
          if (pending_bytes > config_.max_buffered_bytes) {
            if (!recover("live buffer capacity exceeded")) {
              return;
            }
            continue;
          }
          pending.emplace_back(std::move(batch));
        } else if (!awaiting_snapshot && !index_.apply(batch)) {
          if (!recover("live sequence gap or unresolved content")) {
            return;
          }
          continue;
        }
      }
      if (items[2].revents & ZMQ_POLLIN) {
        if (!receive_frames(replay.get(), config_.max_message_bytes, &frames)) {
          return;
        }
        if (!in_flight) {
          continue;
        }
        const std::string sentinel(8, static_cast<char>(0xff));
        if (frames.size() != 3 || !frames[0].empty()) {
          if (!recover("invalid snapshot reply envelope")) {
            return;
          }
          continue;
        }
        if (frames[1] != sentinel) {
          KVEventBatch batch;
          if (snapshot ||
              !decode_kv_batch(frames[1],
                               frames[2],
                               source_.dp_rank,
                               source_.block_size,
                               &batch) ||
              !is_snapshot(batch)) {
            if (!recover("expected one full xLLM snapshot")) {
              return;
            }
            continue;
          }
          snapshot = std::move(batch);
          continue;
        }
        if (!frames[2].empty() || !snapshot ||
            !index_.restore(*snapshot, pending)) {
          if (!recover("incomplete or invalid snapshot")) {
            return;
          }
          continue;
        }
        pending.clear();
        pending_bytes = 0;
        snapshot.reset();
        in_flight = false;
        awaiting_snapshot = false;
        next_request = Clock::now() +
                       std::chrono::milliseconds(config_.resync_interval_ms);
      }
    }
  }

  KVDirectoryConfig config_;
  std::atomic<bool> stopped_{false};
  std::thread thread_;
};

KVDirectory::KVDirectory(KVDirectoryConfig config)
    : config_(std::move(config)) {
  auto sources = std::move(config_.sources);
  streams_.reserve(sources.size());
  for (auto& source : sources) {
    streams_.emplace_back(std::make_shared<Stream>(std::move(source), config_));
  }
}

KVDirectory::~KVDirectory() { stop(); }

void KVDirectory::start(InstanceProvider provider) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (stopped_.load() || started_.exchange(true)) {
    return;
  }
  for (const auto& stream : streams_) {
    stream->start();
  }
  if (config_.discovery.enabled) {
    CHECK(provider) << "KV discovery requires an instance provider";
    discovery_thread_ = std::thread(
        [this, provider = std::move(provider)] { discover(provider); });
  }
}

void KVDirectory::stop() {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  stopped_.store(true);
  wakeup_.notify_all();
  if (discovery_thread_.joinable()) {
    discovery_thread_.join();
  }
  // Signal all receivers before joining them.
  for (const auto& stream : streams_) {
    stream->request_stop();
  }
  for (const auto& stream : streams_) {
    stream->stop();
  }
}

void KVDirectory::discover(const InstanceProvider& provider) {
  while (!stopped_.load()) {
    reconcile(provider());
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wakeup_.wait_for(
        lock,
        std::chrono::milliseconds(config_.discovery.refresh_interval_ms),
        [this] { return stopped_.load(); });
  }
}

void KVDirectory::reconcile(const std::vector<KVEventInstance>& instances) {
  const auto& discovery = config_.discovery;
  std::vector<KVEventInstance> ordered = instances;
  std::sort(
      ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return left.name < right.name;
      });
  std::vector<KVEventSource> desired;
  desired.reserve(config_.max_streams);
  for (const auto& instance : ordered) {
    const std::string address = "tcp://" + instance.name;
    if (!connectable_endpoint(address) || instance.dp_size < 1 ||
        static_cast<size_t>(instance.dp_size) >
            config_.max_streams - desired.size()) {
      LOG_EVERY_N(ERROR, 60)
          << "Cannot discover KV endpoints for " << instance.name
          << ": invalid address/DP size or stream limit exceeded";
      continue;
    }
    const size_t delimiter = address.rfind(':');
    const std::string host = address.substr(0, delimiter + 1);
    uint32_t port = 0;
    for (size_t index = delimiter + 1; index < address.size(); ++index) {
      port = port * 10 + static_cast<uint32_t>(address[index] - '0');
    }
    const uint64_t last_rank = static_cast<uint64_t>(instance.dp_size - 1);
    if (port + discovery.event_port_offset + last_rank > 65535 ||
        port + discovery.replay_port_offset + last_rank > 65535) {
      LOG_EVERY_N(ERROR, 60)
          << "KV discovery port range overflows for " << instance.name;
      continue;
    }
    for (int32_t rank = 0; rank < instance.dp_size; ++rank) {
      KVEventSource source;
      source.instance_name = instance.name;
      source.incarnation_id = instance.incarnation_id;
      source.register_ts_ms = instance.register_ts_ms;
      source.dp_rank = rank;
      source.block_size = discovery.block_size;
      source.hash_seed = discovery.hash_seed;
      source.topic = discovery.topic;
      source.endpoint =
          host + std::to_string(port + discovery.event_port_offset + rank);
      source.replay_endpoint =
          host + std::to_string(port + discovery.replay_port_offset + rank);
      desired.emplace_back(std::move(source));
    }
  }
  // Ambiguous endpoints cannot be attributed safely to either instance/DP.
  std::unordered_map<std::string, size_t> endpoint_counts;
  for (const auto& source : desired) {
    ++endpoint_counts[source.endpoint];
    ++endpoint_counts[source.replay_endpoint];
  }
  std::unordered_set<std::string> ambiguous_instances;
  for (const auto& source : desired) {
    if (endpoint_counts[source.endpoint] > 1 ||
        endpoint_counts[source.replay_endpoint] > 1) {
      ambiguous_instances.insert(source.instance_name);
    }
  }
  if (!ambiguous_instances.empty()) {
    LOG_EVERY_N(ERROR, 60)
        << "KV discovery excluded instances with overlapping port ranges";
  }
  std::vector<std::shared_ptr<Stream>> previous;
  {
    std::shared_lock<std::shared_mutex> lock(streams_mutex_);
    previous = streams_;
  }
  std::vector<std::shared_ptr<Stream>> next;
  next.reserve(desired.size());
  for (auto& source : desired) {
    if (ambiguous_instances.count(source.instance_name) != 0) {
      continue;
    }
    const auto existing =
        std::find_if(previous.begin(), previous.end(), [&](const auto& stream) {
          return stream && same_source(stream->source_, source);
        });
    if (existing != previous.end()) {
      next.emplace_back(std::move(*existing));
      continue;
    }
    auto stream = std::make_shared<Stream>(std::move(source), config_);
    LOG(INFO) << "Subscribing KV stream " << stream->source_.instance_name
              << "/" << stream->source_.dp_rank << " at "
              << stream->source_.endpoint;
    stream->start();
    next.emplace_back(std::move(stream));
  }
  {
    std::unique_lock<std::shared_mutex> lock(streams_mutex_);
    for (const auto& stream : previous) {
      if (stream) {
        stream->request_stop();
      }
    }
    streams_ = std::move(next);
  }
  // Retired streams are absent from queries before their sockets are joined.
  for (const auto& stream : previous) {
    if (stream) {
      LOG(INFO) << "Removing KV stream " << stream->source_.instance_name << "/"
                << stream->source_.dp_rank;
      stream->stop();
    }
  }
}

std::vector<KVDirectoryMatch> KVDirectory::match(
    const std::vector<int32_t>& token_ids) const {
  std::vector<KVDirectoryMatch> result;
  std::vector<std::shared_ptr<Stream>> streams;
  {
    std::shared_lock<std::shared_mutex> lock(streams_mutex_);
    streams = streams_;
  }
  result.reserve(streams.size());
  for (const auto& stream : streams) {
    const auto& source = stream->source_;
    result.emplace_back(KVDirectoryMatch{source.instance_name,
                                         source.dp_rank,
                                         stream->match(token_ids),
                                         source.incarnation_id,
                                         source.register_ts_ms});
  }
  return result;
}

std::vector<KVEventInstance> make_kv_event_instances(
    const std::vector<InstanceView>& instances) {
  std::vector<KVEventInstance> result;
  result.reserve(instances.size());
  for (const auto& view : instances) {
    const auto& info = *view.info;
    if (info.backend_type != "xllm" ||
        view.runtime_state == InstanceRuntimeState::SUSPECT ||
        info.type == InstanceType::DECODE ||
        (info.type == InstanceType::MIX &&
         view.role != InstanceType::PREFILL)) {
      continue;
    }
    result.emplace_back(KVEventInstance{
        info.name, info.incarnation_id, info.register_ts_ms, info.dp_size});
  }
  return result;
}

}  // namespace xllm_service
