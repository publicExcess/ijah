// gguf_logits: full-vocabulary f32 logits of a GGUF through llama.cpp (libllama), teacher-forced over token windows,
// for tools/kld_gguf.sh (lab splashcmp, 23 Oct). Every position's logits, in order, written as raw f32 [T][V].
//   gguf_logits MODEL.gguf IN_DIR OUT_DIR NAME...    IN_DIR/NAME.tok = int32 token ids; writes OUT_DIR/NAME.f32
// One context per window (fresh state), n_ctx = the window length rounded up, batches of 2048 tokens (every token an
// output), all layers on Metal, llama.cpp's defaults otherwise (f16 KV, flash attention auto).
// Build: clang++ -std=c++17 -O2 -I/opt/homebrew/include -I/opt/homebrew/opt/llama.cpp/include tools/gguf_logits.cpp \
//   -L/opt/homebrew/opt/llama.cpp/lib -lllama -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -o build-gguf/gguf_logits
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: gguf_logits MODEL.gguf IN_DIR OUT_DIR NAME...\n");
    return 2;
  }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) { fprintf(stderr, "load failed\n"); return 1; }
  const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
  const int NB = 2048;
  for (int a = 4; a < argc; a++) {
    const std::string name = argv[a];
    FILE *fi = fopen((std::string(argv[2]) + "/" + name + ".tok").c_str(), "rb");
    if (!fi) { fprintf(stderr, "%s: no tokens\n", name.c_str()); return 1; }
    std::vector<int32_t> tok;
    int32_t t;
    while (fread(&t, 4, 1, fi) == 1) tok.push_back(t);
    fclose(fi);
    const int T = (int)tok.size();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)((T + 255) / 256 * 256);
    cp.n_batch = NB;
    cp.n_seq_max = 1;
    cp.no_perf = true;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "%s: context failed\n", name.c_str()); return 1; }
    const std::string out = std::string(argv[3]) + "/" + name + ".f32";
    FILE *fo = fopen((out + ".part").c_str(), "wb");
    llama_batch b = llama_batch_init(NB, 0, 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < T; c += NB) {
      const int n = std::min(NB, T - c);
      b.n_tokens = n;
      for (int i = 0; i < n; i++) {
        b.token[i] = tok[c + i];
        b.pos[i] = c + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
      }
      if (llama_decode(ctx, b) != 0) { fprintf(stderr, "%s: decode failed at %d\n", name.c_str(), c); return 1; }
      for (int i = 0; i < n; i++) {
        const float *l = llama_get_logits_ith(ctx, i);
        if (!l || fwrite(l, 4, V, fo) != (size_t)V) { fprintf(stderr, "%s: logits/write failed\n", name.c_str()); return 1; }
      }
    }
    fclose(fo);
    rename((out + ".part").c_str(), out.c_str());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s: %d tokens, vocab %d, %.1f s (%.0f tok/s)\n", name.c_str(), T, V, s, T / s);
    fflush(stdout);
    llama_batch_free(b);
    llama_free(ctx);
  }
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
