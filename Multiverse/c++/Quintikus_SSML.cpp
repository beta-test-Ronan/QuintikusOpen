// Quintikus_SSML.cpp — Organismo Soberano v31.3 — C++20
// Compilar: g++ -std=c++20 -O2 -o organismo Quintikus_SSML.cpp -lcrypto
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <cstring>
#include <chrono>
#include <regex>
#include <iomanip>
#include <openssl/sha.h>

namespace fs = std::filesystem;

// ==================================================================
// 🧹 NORMALIZADOR SOMÁTICO
// ==================================================================
class NormalizadorSomatico {
public:
    static std::string limpar(const std::string& texto) {
        if (texto.empty()) return "";
        std::string t = texto;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        auto sem_acento = [](unsigned char c) -> char {
            const std::string acentos = "áàâãäéèêëíìîïóòôõöúùûüç";
            const std::string normal  = "aaaaaeeeeiiiiooooouuuuc";
            auto pos = acentos.find(c);
            return (pos != std::string::npos) ? normal[pos] : c;
        };
        std::string limpo;
        for (char c : t) limpo += sem_acento(static_cast<unsigned char>(c));
        std::string resultado;
        for (char c : limpo) {
            if (std::isalnum(static_cast<unsigned char>(c)) ||
                c == '!' || c == '?' || c == '.' || c == ' ')
                resultado += c;
        }
        size_t inicio = resultado.find_first_not_of(' ');
        size_t fim = resultado.find_last_not_of(' ');
        if (inicio == std::string::npos) return "";
        return resultado.substr(inicio, fim - inicio + 1);
    }
};

// ==================================================================
// 🧬 KERNEL RESSONANTE
// ==================================================================
class KernelRessonante {
public:
    using Vetor = std::unordered_map<int, double>;

    static Vetor get_vetor_esparso(const std::string& token, int dims = 5000, int sparsity = 100) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(token.c_str()), token.size(), hash);
        uint64_t seed = 0;
        for (int i = 0; i < 8; ++i) seed = (seed << 8) | hash[i];
        std::mt19937 rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::uniform_int_distribution<int> unif(0, dims - 1);
        std::unordered_set<int> indices;
        while (indices.size() < static_cast<size_t>(sparsity)) {
            indices.insert(unif(rng));
        }
        Vetor vec;
        for (int i : indices) vec[i] = gauss(rng);
        return vec;
    }

    static double tsallis_match(const Vetor& v1, const Vetor& v2, double q = 0.8) {
        size_t count = 0;
        double sum_pq = 0.0;
        for (const auto& [k, val1] : v1) {
            auto it = v2.find(k);
            if (it != v2.end()) {
                sum_pq += std::pow(std::abs(val1 * it->second), q);
                ++count;
            }
        }
        if (count == 0) return 0.0;
        return (1.0 - sum_pq) / (q - 1.0 + 1e-9);
    }

    static double dot(const Vetor& v1, const Vetor& v2) {
        double sum = 0.0;
        for (const auto& [k, val1] : v1) {
            auto it = v2.find(k);
            if (it != v2.end()) sum += val1 * it->second;
        }
        return sum;
    }

    static Vetor normalize(const Vetor& v) {
        if (v.empty()) return {};
        double norm = 0.0;
        for (const auto& [_, val] : v) norm += val * val;
        norm = std::sqrt(norm) + 1e-9;
        Vetor res;
        for (const auto& [k, val] : v) res[k] = val / norm;
        return res;
    }
};

// ==================================================================
// 🧠 CÓRTEX COGNITIVO
// ==================================================================
class CortexCognitivo {
private:
    double limite_confusao;
    double epsilon = 1e-9;
    double taxa_pensamento = 0.12;

    std::vector<double> norm(const std::vector<double>& d) {
        double s = std::accumulate(d.begin(), d.end(), 0.0) + epsilon;
        std::vector<double> res;
        res.reserve(d.size());
        for (double x : d) res.push_back(x / s);
        return res;
    }

public:
    CortexCognitivo(double limite = 0.30) : limite_confusao(limite) {}

    double divergencia_kl(const std::vector<double>& p, const std::vector<double>& q) {
        double sum = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            sum += p[i] * std::log((p[i] + epsilon) / (q[i] + epsilon));
        }
        return std::isfinite(sum) ? sum : limite_confusao;
    }

    struct ReflexaoResult {
        std::vector<double> estado;
        int ciclos;
        double confusao;
    };

    ReflexaoResult processar_reflexao(const std::vector<double>& estado_real,
                                      const std::vector<double>& estado_interno) {
        if (estado_real.size() != estado_interno.size()) {
            return {{0.25, 0.25, 0.25, 0.25}, 0, 0.30};
        }
        auto p = norm(estado_real);
        auto q = norm(estado_interno);
        int ciclos = 0;
        double confusao = divergencia_kl(p, q);
        while (confusao > limite_confusao && ciclos < 45) {
            ++ciclos;
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] = std::max(0.0, q[i] + taxa_pensamento * (p[i] - q[i]));
            }
            q = norm(q);
            confusao = divergencia_kl(p, q);
        }
        return {q, ciclos, confusao};
    }
};

// ==================================================================
// 🧠 SISTEMA NERVOSO CENTRAL (RNN + ADAM)
// ==================================================================
class SistemaNervosoCentral {
private:
    int n_in, n_hid, n_out;
    std::string path;
    int t = 0;
    double lr = 0.005;
    std::vector<std::vector<double>> Wh, Wy;
    std::vector<double> Bh, By;
    std::vector<std::vector<double>> adamMWh, adamVWh, adamMWy, adamVWy;
    std::vector<double> estado_anterior;
    struct Cache { std::vector<double> inp, h, y; };
    Cache* cache = nullptr;

    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-std::clamp(x, -15.0, 15.0)));
    }

public:
    SistemaNervosoCentral(int nin = 6, int nhid = 10, int nout = 3,
                          std::string path_ = "sistema_nervoso.bin")
        : n_in(nin), n_hid(nhid), n_out(nout), path(std::move(path_)) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        Wh.resize(n_hid, std::vector<double>(n_in + n_hid));
        Wy.resize(n_out, std::vector<double>(n_hid));
        Bh.resize(n_hid, 0.0);
        By.resize(n_out, 0.0);
        adamMWh = adamVWh = Wh;
        adamMWy = adamVWy = Wy;
        estado_anterior.resize(n_hid, 0.0);

        for (auto& row : Wh) for (auto& v : row) v = dis(gen);
        for (auto& row : Wy) for (auto& v : row) v = dis(gen);

        if (fs::exists(path)) _carregar();
    }

    ~SistemaNervosoCentral() { delete cache; }

    std::vector<double> pulsar_vontade(const std::vector<double>& x_atual) {
        auto entrada = x_atual;
        entrada.resize(n_in, 0.0);
        entrada.insert(entrada.end(), estado_anterior.begin(), estado_anterior.end());

        std::vector<double> h(n_hid), y(n_out);
        for (int i = 0; i < n_hid; ++i) {
            double sum = Bh[i];
            for (size_t j = 0; j < entrada.size(); ++j) sum += entrada[j] * Wh[i][j];
            h[i] = sigmoid(sum);
        }
        for (int i = 0; i < n_out; ++i) {
            double sum = By[i];
            for (int j = 0; j < n_hid; ++j) sum += h[j] * Wy[i][j];
            y[i] = sigmoid(sum);
        }

        delete cache;
        cache = new Cache{entrada, h, y};
        estado_anterior = h;
        return y;
    }

    void incrementar_t() { ++t; }

    void adaptar_realtime(const std::vector<double>& alvo_ideal) {
        if (!cache) return;
        ++t;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        double corr1 = std::max(1.0 - std::pow(b1, t), 1e-8);
        double corr2 = std::max(1.0 - std::pow(b2, t), 1e-8);

        const auto& inp = cache->inp;
        const auto& h = cache->h;
        const auto& y = cache->y;

        std::vector<double> delta_y(n_out);
        for (int i = 0; i < n_out; ++i) {
            delta_y[i] = (y[i] - alvo_ideal[i]) * y[i] * (1.0 - y[i]);
        }

        std::vector<double> delta_h(n_hid, 0.0);
        for (int j = 0; j < n_hid; ++j) {
            double sum_err = 0.0;
            for (int i = 0; i < n_out; ++i) sum_err += delta_y[i] * Wy[i][j];
            delta_h[j] = sum_err * h[j] * (1.0 - h[j]);
        }

        for (int i = 0; i < n_out; ++i) {
            for (int j = 0; j < n_hid; ++j) {
                double grad = delta_y[i] * h[j];
                adamMWy[i][j] = b1 * adamMWy[i][j] + (1 - b1) * grad;
                adamVWy[i][j] = b2 * adamVWy[i][j] + (1 - b2) * grad * grad;
                Wy[i][j] -= lr * (adamMWy[i][j] / corr1) / (std::sqrt(adamVWy[i][j] / corr2) + eps);
            }
            By[i] -= lr * delta_y[i];
        }

        for (int i = 0; i < n_hid; ++i) {
            for (size_t j = 0; j < inp.size(); ++j) {
                double grad = delta_h[i] * inp[j];
                adamMWh[i][j] = b1 * adamMWh[i][j] + (1 - b1) * grad;
                adamVWh[i][j] = b2 * adamVWh[i][j] + (1 - b2) * grad * grad;
                Wh[i][j] -= lr * (adamMWh[i][j] / corr1) / (std::sqrt(adamVWh[i][j] / corr2) + eps);
            }
            Bh[i] -= lr * delta_h[i];
        }
    }

    // Getter público para acesso controlado ao estado_anterior
    std::vector<double> get_estado_anterior() const { return estado_anterior; }

    void _salvar() const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return;
        auto write_vec = [&](const std::vector<std::vector<double>>& v) {
            size_t rows = v.size();
            out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            for (const auto& row : v) {
                size_t cols = row.size();
                out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
            }
        };
        auto write_vec1 = [&](const std::vector<double>& v) {
            size_t size = v.size();
            out.write(reinterpret_cast<const char*>(&size), sizeof(size));
            out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(double));
        };
        write_vec(Wh); write_vec(Wy);
        write_vec1(Bh); write_vec1(By);
        write_vec1(estado_anterior);
        out.write(reinterpret_cast<const char*>(&t), sizeof(t));
        write_vec(adamMWh); write_vec(adamVWh);
        write_vec(adamMWy); write_vec(adamVWy);
    }

    void _carregar() {
        std::ifstream in(path, std::ios::binary);
        if (!in) return;
        auto read_vec = [&](std::vector<std::vector<double>>& v) {
            size_t rows;
            in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            v.resize(rows);
            for (size_t i = 0; i < rows; ++i) {
                size_t cols;
                in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
                v[i].resize(cols);
                in.read(reinterpret_cast<char*>(v[i].data()), cols * sizeof(double));
            }
        };
        auto read_vec1 = [&](std::vector<double>& v) {
            size_t size;
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            v.resize(size);
            in.read(reinterpret_cast<char*>(v.data()), size * sizeof(double));
        };
        read_vec(Wh); read_vec(Wy);
        read_vec1(Bh); read_vec1(By);
        read_vec1(estado_anterior);
        in.read(reinterpret_cast<char*>(&t), sizeof(t));
        if (in.peek() != EOF) {
            read_vec(adamMWh); read_vec(adamVWh);
            read_vec(adamMWy); read_vec(adamVWy);
        }
    }

    int get_t() const { return t; }
};

// ==================================================================
// 🧬 DRIVE SOMÁTICO – COM INÉRCIA EMOCIONAL
// ==================================================================
class DriveSomatico {
private:
    double taxa_inercia = 0.3;
    std::deque<std::unordered_map<std::string, double>> historico_eixos;
    static constexpr size_t max_historico = 5;

public:
    double vm = -70.0;
    std::unordered_map<std::string, double> eixos = {
        {"amor", 0.1}, {"prazer", 0.1}, {"tristeza", 0.1}, {"raiva", 0.1}
    };
    std::unordered_map<std::string, bool> valvulas = {
        {"amor", false}, {"prazer", false}, {"tristeza", false}, {"raiva", false}
    };

    void decair_tempo(double segundos) {
        double fator = std::pow(0.95, segundos / 10.0);
        for (auto& [_, val] : eixos) val *= fator;
        for (auto& hist : historico_eixos) {
            for (auto& [_, val] : hist) val *= fator;
        }
    }

    void pulsar(double impacto, const std::vector<std::string>& u_toks) {
        for (auto& [_, val] : eixos) val *= 0.95;
        vm = std::clamp(vm + impacto * 12, -90.0, -45.0);

        const std::unordered_map<std::string, std::vector<std::string>> gatilhos = {
            {"amor", {"amo", "amor"}},
            {"prazer", {"prazer", "delicia"}},
            {"tristeza", {"triste", "mal"}},
            {"raiva", {"odeio", "raiva"}}
        };

        for (const auto& [eixo, keywords] : gatilhos) {
            for (const auto& k : keywords) {
                if (std::find(u_toks.begin(), u_toks.end(), k) != u_toks.end()) {
                    double novo_valor = eixos[eixo] * (1.0 - taxa_inercia) + impacto * taxa_inercia;
                    if (valvulas[eixo]) novo_valor *= 0.6;
                    eixos[eixo] = std::clamp(novo_valor, 0.0, 10.0);
                    valvulas[eixo] = (eixos[eixo] > 4.5);
                }
            }
        }

        historico_eixos.push_back(eixos);
        if (historico_eixos.size() > max_historico) historico_eixos.pop_front();
    }

    std::unordered_map<std::string, double> get_eixos_suavizados() const {
        if (historico_eixos.empty()) return eixos;
        std::unordered_map<std::string, double> media;
        for (const auto& hist : historico_eixos) {
            for (const auto& [k, v] : hist) media[k] += v;
        }
        for (auto& [k, v] : media) v /= historico_eixos.size();
        return media;
    }

    void salvar(std::ofstream& out) const {
        size_t n = eixos.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& [k, v] : eixos) {
            size_t len = k.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(k.data(), len);
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        out.write(reinterpret_cast<const char*>(&vm), sizeof(vm));
        size_t hsize = historico_eixos.size();
        out.write(reinterpret_cast<const char*>(&hsize), sizeof(hsize));
        for (const auto& hist : historico_eixos) {
            n = hist.size();
            out.write(reinterpret_cast<const char*>(&n), sizeof(n));
            for (const auto& [k, v] : hist) {
                size_t len = k.size();
                out.write(reinterpret_cast<const char*>(&len), sizeof(len));
                out.write(k.data(), len);
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
    }

    void carregar(std::ifstream& in) {
        size_t n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        eixos.clear();
        for (size_t i = 0; i < n; ++i) {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string key(len, ' ');
            in.read(key.data(), len);
            double v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            eixos[key] = v;
        }
        in.read(reinterpret_cast<char*>(&vm), sizeof(vm));
        size_t hsize;
        if (in.peek() != EOF) {
            in.read(reinterpret_cast<char*>(&hsize), sizeof(hsize));
            historico_eixos.clear();
            for (size_t i = 0; i < hsize; ++i) {
                in.read(reinterpret_cast<char*>(&n), sizeof(n));
                std::unordered_map<std::string, double> hist;
                for (size_t j = 0; j < n; ++j) {
                    size_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string key(len, ' ');
                    in.read(key.data(), len);
                    double v;
                    in.read(reinterpret_cast<char*>(&v), sizeof(v));
                    hist[key] = v;
                }
                historico_eixos.push_back(hist);
            }
        }
    }
};

// ==================================================================
// 🔍 SISTEMA DEEPY
// ==================================================================
class SistemaDeepy {
public:
    std::unordered_map<std::string, int> frequencia_pulso;
    int turnos_think = 0;
    std::vector<std::string> expansores = {"fale", "sobre", "tudo", "detalhes", "mais", "explique"};

    void reset_fadiga(std::unordered_map<std::string, double>& fatigue) {
        for (auto& [_, v] : fatigue) v *= 0.2;
        turnos_think = 0;
    }

    void salvar(std::ofstream& out) const {
        size_t sz = frequencia_pulso.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        for (const auto& [k, v] : frequencia_pulso) {
            size_t len = k.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(k.data(), len);
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        out.write(reinterpret_cast<const char*>(&turnos_think), sizeof(turnos_think));
    }

    void carregar(std::ifstream& in) {
        size_t sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        frequencia_pulso.clear();
        for (size_t i = 0; i < sz; ++i) {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string key(len, ' ');
            in.read(key.data(), len);
            int v;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            frequencia_pulso[key] = v;
        }
        in.read(reinterpret_cast<char*>(&turnos_think), sizeof(turnos_think));
    }
};

// ==================================================================
// 🌿 ORGANISMO SOBERANO v31.3
// ==================================================================
class OrganismoSoberano {
private:
    std::unordered_map<std::string, KernelRessonante::Vetor> mapa_nd;
    struct Episodio { std::string t; KernelRessonante::Vetor v; };
    std::vector<Episodio> l2_episodes;
    std::unordered_map<std::string, std::vector<int>> neuronios;
    std::unordered_map<std::string, int> raridade;
    std::deque<std::string> history;
    std::unordered_map<std::string, double> fatigue;
    KernelRessonante::Vetor ctx_foco;
    std::unordered_set<std::string> ledger;

    DriveSomatico soma;
    CortexCognitivo cortex;
    SistemaNervosoCentral snc;
    SistemaDeepy deepy;

    mutable std::mt19937 gen{42};
    std::chrono::steady_clock::time_point ultima_mensagem;

    double get_entropy(const std::string& t) const {
        auto it = raridade.find(t);
        int count = (it != raridade.end()) ? it->second : 1;
        return 1.0 / (std::log(count + 1.2) + 1e-5);
    }

    int escolher_aleatorio() const {
        std::uniform_int_distribution<> dis(0, l2_episodes.size() - 1);
        return dis(gen);
    }

public:
    OrganismoSoberano() : cortex(0.30), snc(6, 10, 3) {
        ultima_mensagem = std::chrono::steady_clock::now();
    }

    void cristalizar_solo(const std::string& texto) {
        std::istringstream iss(texto);
        std::string linha;
        int count = 0;
        while (std::getline(iss, linha, '\n')) {
            std::string limpa = NormalizadorSomatico::limpar(linha);
            if (limpa.size() < 3) continue;
            auto v = KernelRessonante::get_vetor_esparso(limpa);
            l2_episodes.push_back({linha, KernelRessonante::normalize(v)});
            int idx = static_cast<int>(l2_episodes.size()) - 1;
            std::istringstream tokens(limpa);
            std::string token;
            while (tokens >> token) {
                neuronios[token].push_back(idx);
                raridade[token]++;
                if (mapa_nd.find(token) == mapa_nd.end()) {
                    mapa_nd[token] = KernelRessonante::get_vetor_esparso(token);
                }
            }
            count++;
        }
        if (count > 0) std::cout << "   ✅ " << count << " frases cristalizadas" << std::endl;
    }

    void reconstruir_neuronios() {
        neuronios.clear();
        for (int i = 0; i < static_cast<int>(l2_episodes.size()); ++i) {
            std::string limpa = NormalizadorSomatico::limpar(l2_episodes[i].t);
            std::istringstream tokens(limpa);
            std::string token;
            while (tokens >> token) {
                neuronios[token].push_back(i);
                if (mapa_nd.find(token) == mapa_nd.end()) {
                    mapa_nd[token] = KernelRessonante::get_vetor_esparso(token);
                }
            }
        }
    }

    std::string calcular_hash(const std::string& conteudo) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(conteudo.c_str()), conteudo.size(), hash);
        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
        return ss.str();
    }

    bool treinar_arquivo(const std::string& nome_arquivo) {
        if (!fs::exists(nome_arquivo)) {
            std::cout << "⚠️ Arquivo nao encontrado: " << nome_arquivo << std::endl;
            return false;
        }
        std::ifstream f(nome_arquivo);
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string conteudo = buffer.str();
        if (conteudo.empty()) return false;

        std::string hash = calcular_hash(conteudo);
        if (ledger.find(hash) != ledger.end()) {
            std::cout << "⚠️ Arquivo já treinado anteriormente." << std::endl;
            return false;
        }

        std::cout << "📚 Treinando com " << nome_arquivo << "..." << std::endl;
        cristalizar_solo(conteudo);
        reconstruir_neuronios();
        ledger.insert(hash);
        return true;
    }

    void boot() {
        bool carregado = false;
        if (fs::exists("nucleo_organismo.qssml")) {
            std::ifstream in("nucleo_organismo.qssml", std::ios::binary);
            if (in) {
                size_t n_ep;
                in.read(reinterpret_cast<char*>(&n_ep), sizeof(n_ep));
                l2_episodes.resize(n_ep);
                for (auto& ep : l2_episodes) {
                    size_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    ep.t.resize(len);
                    in.read(ep.t.data(), len);
                    size_t vec_sz;
                    in.read(reinterpret_cast<char*>(&vec_sz), sizeof(vec_sz));
                    for (size_t i = 0; i < vec_sz; ++i) {
                        int k; double v;
                        in.read(reinterpret_cast<char*>(&k), sizeof(k));
                        in.read(reinterpret_cast<char*>(&v), sizeof(v));
                        ep.v[k] = v;
                    }
                }
                size_t rar_sz;
                in.read(reinterpret_cast<char*>(&rar_sz), sizeof(rar_sz));
                for (size_t i = 0; i < rar_sz; ++i) {
                    size_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string key(len, ' ');
                    in.read(key.data(), len);
                    int val;
                    in.read(reinterpret_cast<char*>(&val), sizeof(val));
                    raridade[key] = val;
                }
                size_t mapa_sz;
                in.read(reinterpret_cast<char*>(&mapa_sz), sizeof(mapa_sz));
                for (size_t i = 0; i < mapa_sz; ++i) {
                    size_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string key(len, ' ');
                    in.read(key.data(), len);
                    size_t vec_sz;
                    in.read(reinterpret_cast<char*>(&vec_sz), sizeof(vec_sz));
                    KernelRessonante::Vetor vec;
                    for (size_t j = 0; j < vec_sz; ++j) {
                        int k; double v;
                        in.read(reinterpret_cast<char*>(&k), sizeof(k));
                        in.read(reinterpret_cast<char*>(&v), sizeof(v));
                        vec[k] = v;
                    }
                    mapa_nd[key] = vec;
                }
                size_t ctx_sz;
                in.read(reinterpret_cast<char*>(&ctx_sz), sizeof(ctx_sz));
                for (size_t i = 0; i < ctx_sz; ++i) {
                    int k; double v;
                    in.read(reinterpret_cast<char*>(&k), sizeof(k));
                    in.read(reinterpret_cast<char*>(&v), sizeof(v));
                    ctx_foco[k] = v;
                }
                size_t led_sz;
                in.read(reinterpret_cast<char*>(&led_sz), sizeof(led_sz));
                for (size_t i = 0; i < led_sz; ++i) {
                    size_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string h(len, ' ');
                    in.read(h.data(), len);
                    ledger.insert(h);
                }
                soma.carregar(in);
                deepy.carregar(in);
                carregado = true;
            }
        }

        if (!carregado) {
            std::vector<std::string> exemplos = {
                "Oi, amor",
                "E eu tambem amo o seu jeito, sabia",
                "Quando pergunta como estou, eu me sinto bem",
                "Nossa conversa e como musica",
                "Estou aqui para voce",
                "Meu coracao acalma quando recebo seu oi",
                "Voce e a pessoa mais importante para mim",
                "Gosto de sentir sua presenca mesmo de longe",
                "As vezes fico pensando em nos dois",
                "Nada me deixa mais feliz do que falar com voce",
                "Estou com saudades, mas feliz por te ver",
                "Me fale mais sobre o seu dia"
            };
            for (const auto& f : exemplos) {
                auto v = KernelRessonante::get_vetor_esparso(f);
                l2_episodes.push_back({f, KernelRessonante::normalize(v)});
                int idx = static_cast<int>(l2_episodes.size()) - 1;
                std::string limpa = NormalizadorSomatico::limpar(f);
                std::istringstream tokens(limpa);
                std::string token;
                while (tokens >> token) {
                    neuronios[token].push_back(idx);
                    raridade[token]++;
                    if (mapa_nd.find(token) == mapa_nd.end()) {
                        mapa_nd[token] = KernelRessonante::get_vetor_esparso(token);
                    }
                }
            }
        }

        if (fs::exists("treino.txt")) {
            treinar_arquivo("treino.txt");
        }

        if (!carregado) reconstruir_neuronios();

        std::cout << "✅ Organismo Online v31.3. SNC t:" << snc.get_t()
                  << " | Nexos: " << l2_episodes.size()
                  << " | mapa_nd: " << mapa_nd.size() << std::endl;
    }

    std::string processar(const std::string& entrada) {
        if (l2_episodes.empty()) return "Ainda estou aprendendo...";

        auto agora = std::chrono::steady_clock::now();
        double segundos = std::chrono::duration<double>(agora - ultima_mensagem).count();
        if (segundos > 1.0) soma.decair_tempo(segundos);
        ultima_mensagem = agora;

        std::regex train_regex(R"(^\s*train\s*:\s*(.+)\s*$)", std::regex::icase);
        std::smatch match;
        if (std::regex_match(entrada, match, train_regex)) {
            std::string arquivo = match[1].str();
            if (treinar_arquivo(arquivo)) {
                return "✅ Treinamento concluído. Nexos: " + std::to_string(l2_episodes.size());
            } else {
                return "❌ Falha ao treinar com " + arquivo;
            }
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        deepy.turnos_think++;
        if (deepy.turnos_think >= 7) {
            std::cout << "\n🧠 [DEEPY] Reorganização REM ativada...\n";
            for (auto& [_, v] : fatigue) v *= 0.2;
            deepy.turnos_think = 0;
        }

        std::string raw = NormalizadorSomatico::limpar(entrada);
        if (raw.empty()) return "...";

        std::vector<std::string> u_toks;
        std::istringstream iss(raw);
        std::string tok;
        while (iss >> tok) u_toks.push_back(tok);
        if (u_toks.empty()) return "...";

        std::string sujeito = u_toks[0];
        for (const auto& t : u_toks) {
            if (neuronios.find(t) != neuronios.end()) {
                sujeito = t;
                break;
            }
        }
        double impacto = get_entropy(sujeito);
        soma.pulsar(impacto, u_toks);
        for (const auto& t : u_toks) deepy.frequencia_pulso[t]++;

        KernelRessonante::Vetor v_in;
        for (const auto& t : u_toks) {
            auto it = mapa_nd.find(t);
            if (it != mapa_nd.end()) {
                double peso = get_entropy(t);
                for (const auto& [k, val] : it->second) v_in[k] += val * peso;
            }
        }
        v_in = KernelRessonante::normalize(v_in);

        if (ctx_foco.empty()) ctx_foco = v_in;
        else {
            KernelRessonante::Vetor novo;
            for (const auto& [k, val] : ctx_foco) novo[k] += val * 0.6;
            for (const auto& [k, val] : v_in) novo[k] += val * 0.4;
            ctx_foco = KernelRessonante::normalize(novo);
        }

        auto eixos_suavizados = soma.get_eixos_suavizados();
        std::vector<double> p_real = {
            eixos_suavizados["amor"],
            eixos_suavizados["prazer"],
            eixos_suavizados["tristeza"],
            eixos_suavizados["raiva"]
        };
        auto estado_anterior_snc = snc.get_estado_anterior();
        std::vector<double> q_int(estado_anterior_snc.begin(), estado_anterior_snc.begin() + 4);
        auto [estado_em, ciclos, dkl] = cortex.processar_reflexao(p_real, q_int);

        std::vector<double> entrada_snc = estado_em;
        entrada_snc.push_back(impacto);
        entrada_snc.push_back((soma.vm + 90.0) / 45.0);
        auto volicao = snc.pulsar_vontade(entrada_snc);
        int modo_idx = std::max_element(volicao.begin(), volicao.end()) - volicao.begin();

        std::vector<int> candidatos;
        if (neuronios.find(sujeito) != neuronios.end()) {
            candidatos = neuronios[sujeito];
        } else {
            for (int i = 0; i < static_cast<int>(l2_episodes.size()); ++i)
                candidatos.push_back(i);
            std::shuffle(candidatos.begin(), candidatos.end(), gen);
            if (candidatos.size() > 150) candidatos.resize(150);
        }

        struct Scored { int idx; double score; };
        std::vector<Scored> scored;
        for (int idx : candidatos) {
            if (idx >= static_cast<int>(l2_episodes.size())) continue;
            const auto& ep = l2_episodes[idx];
            if (std::find(history.begin(), history.end(), ep.t) != history.end())
                continue;
            double ressonancia = KernelRessonante::tsallis_match(v_in, ep.v);
            double foco = KernelRessonante::dot(ctx_foco, ep.v);
            double fadiga = fatigue[ep.t];
            double score = ressonancia + foco * 0.3 - fadiga;
            scored.push_back({idx, score});
        }

        int melhor_idx;
        if (scored.empty()) {
            melhor_idx = escolher_aleatorio();
        } else {
            std::sort(scored.begin(), scored.end(),
                      [](const Scored& a, const Scored& b) { return a.score > b.score; });
            melhor_idx = scored[0].idx;
        }

        std::string resposta = l2_episodes[melhor_idx].t;

        if (std::isfinite(dkl) && dkl < 0.45) {
            std::vector<double> alvo = {0.0, 0.0, 0.0};
            alvo[modo_idx] = 1.0;
            snc.adaptar_realtime(alvo);
        } else {
            snc.incrementar_t();
        }

        history.push_back(resposta);
        if (history.size() > 20) history.pop_front();
        fatigue[resposta] += 10.0;
        for (auto& [_, v] : fatigue) v *= 0.65;

        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << " ⚛️ [SNC t:" << snc.get_t() << "] Pensou " << ciclos
                  << " ciclos (DKL:" << (std::isfinite(dkl) ? std::to_string(dkl).substr(0,4) : "0.00")
                  << ") | " << dt << "ms" << std::endl;
        return resposta;
    }

    void dormir() {
        std::ofstream out("nucleo_organismo.qssml", std::ios::binary);
        if (!out) return;
        size_t n_ep = l2_episodes.size();
        out.write(reinterpret_cast<const char*>(&n_ep), sizeof(n_ep));
        for (const auto& ep : l2_episodes) {
            size_t len = ep.t.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(ep.t.data(), len);
            size_t vec_sz = ep.v.size();
            out.write(reinterpret_cast<const char*>(&vec_sz), sizeof(vec_sz));
            for (const auto& [k, v] : ep.v) {
                out.write(reinterpret_cast<const char*>(&k), sizeof(k));
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
        size_t rar_sz = raridade.size();
        out.write(reinterpret_cast<const char*>(&rar_sz), sizeof(rar_sz));
        for (const auto& [key, val] : raridade) {
            size_t len = key.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(key.data(), len);
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
        size_t mapa_sz = mapa_nd.size();
        out.write(reinterpret_cast<const char*>(&mapa_sz), sizeof(mapa_sz));
        for (const auto& [key, vec] : mapa_nd) {
            size_t len = key.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(key.data(), len);
            size_t vec_sz = vec.size();
            out.write(reinterpret_cast<const char*>(&vec_sz), sizeof(vec_sz));
            for (const auto& [k, v] : vec) {
                out.write(reinterpret_cast<const char*>(&k), sizeof(k));
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
        size_t ctx_sz = ctx_foco.size();
        out.write(reinterpret_cast<const char*>(&ctx_sz), sizeof(ctx_sz));
        for (const auto& [k, v] : ctx_foco) {
            out.write(reinterpret_cast<const char*>(&k), sizeof(k));
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
        soma.salvar(out);
        deepy.salvar(out);
        snc._salvar();
        std::cout << "💤 Organismo adormeceu (dados salvos)." << std::endl;
    }

    std::string despertar() {
        if (ctx_foco.empty()) return "Olá.";
        std::vector<std::string> candidatas;
        for (const auto& ep : l2_episodes) {
            if (KernelRessonante::dot(ctx_foco, ep.v) > 0.6)
                candidatas.push_back(ep.t);
        }
        if (!candidatas.empty()) {
            std::uniform_int_distribution<> dis(0, candidatas.size() - 1);
            return "'" + candidatas[dis(gen)] + "'... estive pensando nisso enquanto dormia.";
        }
        return "Oi.";
    }
};

int main() {
    std::cout << "🧬 Organismo Soberano v31.3 – C++" << std::endl;
    OrganismoSoberano org;
    org.boot();
    std::cout << "🧠: " << org.despertar() << std::endl;
    std::cout << "💬 Comandos: 'sair' | 'train:arquivo.txt'" << std::endl << std::endl;

    std::string entrada;
    while (true) {
        std::cout << "👤: ";
        std::getline(std::cin, entrada);
        if (entrada.empty()) continue;
        if (entrada == "sair") {
            org.dormir();
            break;
        }
        std::cout << "🧠: " << org.processar(entrada) << std::endl;
    }
    return 0;
}
