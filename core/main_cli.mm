// engine-cli: device info, metallib functions, pack inspection, dispatch-cost bench. Owner: lab-core.
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstring>

#include "core/bench.h"
#include "core/engine.hpp"
#include "core/pack_align.h"

using namespace ne;

static int usage() {
  fprintf(stderr,
          "usage: engine-cli info\n"
          "       engine-cli kernels\n"
          "       engine-cli pack DIR [SUBSTRING]     list tensors (dtype, shape, file)\n"
          "       engine-cli config DIR               print config.json (text_config keys)\n"
          "       engine-cli bench [BACKEND...]       dispatch cost (waits for the machine lock)\n"
          "       engine-cli align-pack SRC DST [--align N] [--dry-run] [--only SUBSTR] [--no-verify]\n"
          "                                           copy a pack with every tensor N-aligned (default 64); aligned\n"
          "                                           files and non-safetensors files become symlinks\n");
  return 64;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    try {
      if (cmd == "info") {
        Device &d = Device::get();
        printf("device        %s\n", d.name.c_str());
        printf("has_nax       %d (Apple10 family)\n", d.has_nax);
        printf("has_metal4    %d\n", d.has_metal4);
        printf("maxBuffer     %.2f GiB\n", d.max_buffer_length / 1073741824.0);
        printf("workingSet    %.2f GiB\n", d.dev.recommendedMaxWorkingSetSize / 1073741824.0);
        printf("backend       %s (auto / NE_BACKEND)\n", backend_name(resolve_backend(Backend::Auto)));
        printf("metallib      %s\n", Pipelines::library_path().c_str());
      } else if (cmd == "kernels") {
        for (auto &n : Pipelines::function_names()) printf("%s\n", n.c_str());
      } else if (cmd == "pack" && argc >= 3) {
        double t0 = now_us();
        auto p = Pack::open(argv[2]);
        double t1 = now_us();
        const char *sub = argc >= 4 ? argv[3] : nullptr;
        for (auto &n : p->names()) {
          if (sub && !strstr(n.c_str(), sub)) continue;
          const TensorInfo *ti = p->info(n);
          printf("%-90s %-24s %12llu %s%s\n", n.c_str(), ti->t.desc().c_str(), (unsigned long long)ti->nbytes,
                 ti->file.substr(ti->file.rfind('/') + 1).c_str(), ti->t.buf ? "" : "  (no GPU view)");
        }
        size_t mis[3] = {};  // tensors whose GPU address is not 4 / 16 / 64-byte aligned
        double misb[3] = {};
        for (auto &n : p->names()) {
          const TensorInfo *ti = p->info(n);
          for (int k = 0; k < 3; k++)
            if (ti->file_offset % (4u << (2 * k))) { mis[k]++; misb[k] += ti->nbytes; }
        }
        fprintf(stderr, "%zu tensors, %zu files, %.1f GB mapped, opened in %.0f ms\n", p->names().size(),
                p->num_files(), p->mapped_bytes() / 1e9, (t1 - t0) / 1e3);
        fprintf(stderr, "misaligned: %zu tensors (%.1f GB) not 4-aligned, %zu (%.1f GB) not 16, %zu (%.1f GB) not 64\n",
                mis[0], misb[0] / 1e9, mis[1], misb[1] / 1e9, mis[2], misb[2] / 1e9);
      } else if (cmd == "config" && argc >= 3) {
        auto p = Pack::open(argv[2]);
        for (auto &kv : p->text_config().obj()) {
          std::string v = kv.second.dump();
          if (v.size() > 100) v = v.substr(0, 97) + "...";
          printf("%-40s %s\n", kv.first.c_str(), v.c_str());
        }
      } else if (cmd == "bench") {
        MachineLock lock("lab-core", "engine-cli bench", 15, true);
        std::vector<Backend> bs;
        for (int i = 2; i < argc; i++) bs.push_back(backend_from_name(argv[i]));
        if (bs.empty()) {
          if (Device::get().has_metal4) bs.push_back(Backend::MTL4);
          bs.insert(bs.end(), {Backend::ICB, Backend::Classic, Backend::Serial});
        }
        printf("%-8s %-8s %-11s %10s %10s %10s\n", "backend", "kernel", "chain", "gpu_us", "naive_us", "enc_us");
        for (Backend b : bs)
          for (const char *k : {"tiny", "read256"})
            for (bool indep : {false, true}) {
              DispatchCost c = bench_dispatch(b, k, 256, 2048, 7, indep);
              printf("%-8s %-8s %-11s %10.3f %10.3f %10.3f\n", backend_name(b), k, indep ? "independent" : "dependent",
                     c.gpu_us, c.naive_us, c.encode_us);
              fflush(stdout);
            }
      } else if (cmd == "align-pack" && argc >= 4) {
        AlignOptions o;
        for (int i = 4; i < argc; i++) {
          std::string a = argv[i];
          if (a == "--align" && i + 1 < argc) o.align = strtoull(argv[++i], nullptr, 10);
          else if (a == "--dry-run") o.dry_run = true;
          else if (a == "--no-verify") o.verify = false;
          else if (a == "--only" && i + 1 < argc) o.only = argv[++i];
          else return usage();
        }
        double t0 = now_us();
        AlignStats st = align_pack(argv[2], argv[3], o);
        printf("%s: %llu misaligned tensors (%.1f GB); %llu files rewritten (%.1f GB%s), %llu linked; %.1f s\n",
               o.dry_run ? "dry run" : "done", (unsigned long long)st.misaligned_tensors, st.misaligned_bytes / 1e9,
               (unsigned long long)st.files_rewritten, (o.dry_run ? st.bytes_to_write : st.bytes_written) / 1e9,
               o.dry_run ? " to write" : " written", (unsigned long long)st.files_linked, (now_us() - t0) / 1e6);
      } else {
        return usage();
      }
    } catch (const std::exception &e) {
      fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
