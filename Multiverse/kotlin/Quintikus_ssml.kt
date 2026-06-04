// OrganismoSoberano_v31.kt - Quintikus SSML v31.1 - Kotlin (sem warnings)
import java.io.*
import java.security.MessageDigest
import kotlin.math.*
import java.text.Normalizer

// ==================================================================
// 🧹 NORMALIZADOR SOMÁTICO
// ==================================================================
object NormalizadorSomático {
    fun limpar(texto: String?): String {
        if (texto.isNullOrEmpty()) return ""
        var txt = texto.toLowerCase()
        txt = Normalizer.normalize(txt, Normalizer.Form.NFD)
            .replace(Regex("\\p{InCombiningDiacriticalMarks}+"), "")
        txt = txt.replace(Regex("[^a-z0-9!?.\\s]"), "")
        return txt.trim()
    }
}

// ==================================================================
// 🧬 KERNEL RESSONANTE
// ==================================================================
object KernelRessonante {
    fun getVetorEsparso(token: String, dims: Int = 5000, sparsity: Int = 100): Map<Int, Double> {
        val hash = MessageDigest.getInstance("SHA-256")
            .digest(token.toByteArray())
            .joinToString("") { "%02x".format(it) }
        val seed = hash.substring(0, 15).toLong(16)
        val rng = java.util.Random(seed)
        val indices = mutableSetOf<Int>()
        while (indices.size < sparsity) {
            indices.add(rng.nextInt(dims))
        }
        return indices.associateWith { rng.nextGaussian() }
    }

    fun tsallisMatch(v1: Map<Int, Double>, v2: Map<Int, Double>, q: Double = 0.8): Double {
        val keys = v1.keys.intersect(v2.keys)
        if (keys.isEmpty()) return 0.0
        var sumPq = 0.0
        for (k in keys) {
            sumPq += abs(v1[k]!! * v2[k]!!).pow(q)
        }
        return (1.0 - sumPq) / (q - 1.0 + 1e-9)
    }

    fun dot(v1: Map<Int, Double>, v2: Map<Int, Double>): Double {
        val keys = v1.keys.intersect(v2.keys)
        if (keys.isEmpty()) return 0.0
        var sum = 0.0
        for (k in keys) {
            sum += v1[k]!! * v2[k]!!
        }
        return sum
    }

    fun normalize(v: Map<Int, Double>): Map<Int, Double> {
        if (v.isEmpty()) return emptyMap()
        var sum = 0.0
        for (value in v.values) {
            sum += value * value
        }
        val norm = sqrt(sum) + 1e-9
        val result = mutableMapOf<Int, Double>()
        for ((k, value) in v) {
            result[k] = value / norm
        }
        return result
    }
}

// ==================================================================
// 🧠 CÓRTEX COGNITIVO
// ==================================================================
class CortexCognitivo(val limiteConfusao: Double = 0.30) {
    private val epsilon = 1e-9
    private val taxaPensamento = 0.12

    private fun norm(d: List<Double>): List<Double> {
        val s = d.sum() + epsilon
        return d.map { it / s }
    }

    fun divergenciaKL(p: List<Double>, q: List<Double>): Double {
        var sum = 0.0
        for (i in p.indices) {
            sum += p[i] * ln((p[i] + epsilon) / (q[i] + epsilon))
        }
        return if (sum.isFinite()) sum else limiteConfusao
    }

    fun processarReflexao(estadoReal: List<Double>, estadoInterno: List<Double>): Triple<List<Double>, Int, Double> {
        if (estadoReal.size != estadoInterno.size) {
            return Triple(listOf(0.25, 0.25, 0.25, 0.25), 0, 0.30)
        }
        
        val p = norm(estadoReal)
        val q = norm(estadoInterno).toMutableList()
        var ciclos = 0
        var confusao = divergenciaKL(p, q)
        
        while (confusao > limiteConfusao && ciclos < 45) {
            ciclos++
            for (i in q.indices) {
                q[i] = max(0.0, q[i] + taxaPensamento * (p[i] - q[i]))
            }
            val qNorm = norm(q)
            for (i in q.indices) q[i] = qNorm[i]
            confusao = divergenciaKL(p, q)
        }
        
        return Triple(q.toList(), ciclos, confusao)
    }
}

// ==================================================================
// 🧠 SISTEMA NERVOSO CENTRAL (RNN + ADAM)
// ==================================================================
class SistemaNervosoCentral(
    val nIn: Int = 6,
    val nHid: Int = 10,
    val nOut: Int = 3,
    val path: String = "sistema_nervoso.bin"
) : Serializable {
    var t = 0
    val lr = 0.005
    
    var Wh = Array(nHid) { DoubleArray(nIn + nHid) { java.util.Random().nextDouble() * 0.2 - 0.1 } }
    var Wy = Array(nOut) { DoubleArray(nHid) { java.util.Random().nextDouble() * 0.2 - 0.1 } }
    var Bh = DoubleArray(nHid)
    var By = DoubleArray(nOut)
    
    var adamMWh = Array(nHid) { DoubleArray(nIn + nHid) }
    var adamVWh = Array(nHid) { DoubleArray(nIn + nHid) }
    var adamMWy = Array(nOut) { DoubleArray(nHid) }
    var adamVWy = Array(nOut) { DoubleArray(nHid) }
    
    var estadoAnterior = DoubleArray(nHid)
    private var cache: Triple<DoubleArray, DoubleArray, DoubleArray>? = null
    
    init {
        if (File(path).exists()) carregar()
    }
    
    private fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + exp(-x.coerceIn(-15.0, 15.0)))
    }
    
    fun pulsarVontade(xAtual: List<Double>): DoubleArray {
        val entrada = xAtual.take(nIn).toMutableList()
        while (entrada.size < nIn) entrada.add(0.0)
        
        val inp = DoubleArray(entrada.size + estadoAnterior.size)
        for (i in entrada.indices) inp[i] = entrada[i]
        for (i in estadoAnterior.indices) inp[entrada.size + i] = estadoAnterior[i]
        
        val h = DoubleArray(nHid) { i ->
            var sum = Bh[i]
            for (j in inp.indices) sum += inp[j] * Wh[i][j]
            sigmoid(sum)
        }
        
        val y = DoubleArray(nOut) { i ->
            var sum = By[i]
            for (j in h.indices) sum += h[j] * Wy[i][j]
            sigmoid(sum)
        }
        
        cache = Triple(inp, h, y)
        estadoAnterior = h
        return y
    }
    
    fun adaptarRealtime(alvoIdeal: List<Double>) {
        val (inp, h, y) = cache ?: return
        t++
        val b1 = 0.9
        val b2 = 0.999
        val eps = 1e-8
        val corr1 = max(1.0 - b1.pow(t), 1e-8)
        val corr2 = max(1.0 - b2.pow(t), 1e-8)
        
        val deltaY = DoubleArray(nOut) { i ->
            (y[i] - alvoIdeal[i]) * y[i] * (1.0 - y[i])
        }
        
        val deltaH = DoubleArray(nHid) { j ->
            var sumErr = 0.0
            for (i in 0 until nOut) sumErr += deltaY[i] * Wy[i][j]
            sumErr * h[j] * (1.0 - h[j])
        }
        
        for (i in 0 until nOut) {
            for (j in 0 until nHid) {
                val grad = deltaY[i] * h[j]
                adamMWy[i][j] = b1 * adamMWy[i][j] + (1 - b1) * grad
                adamVWy[i][j] = b2 * adamVWy[i][j] + (1 - b2) * (grad * grad)
                Wy[i][j] -= lr * (adamMWy[i][j] / corr1) / (sqrt(abs(adamVWy[i][j]) / corr2) + eps)
            }
            By[i] -= lr * deltaY[i]
        }
        
        for (i in 0 until nHid) {
            for (j in inp.indices) {
                val grad = deltaH[i] * inp[j]
                adamMWh[i][j] = b1 * adamMWh[i][j] + (1 - b1) * grad
                adamVWh[i][j] = b2 * adamVWh[i][j] + (1 - b2) * (grad * grad)
                Wh[i][j] -= lr * (adamMWh[i][j] / corr1) / (sqrt(abs(adamVWh[i][j]) / corr2) + eps)
            }
            Bh[i] -= lr * deltaH[i]
        }
    }
    
    fun salvar() {
        try {
            val estado = SaveState(Wh, Wy, Bh, By, estadoAnterior, t, adamMWh, adamVWh, adamMWy, adamVWy)
            ObjectOutputStream(FileOutputStream(path)).use { it.writeObject(estado) }
        } catch (e: Exception) {
            println("⚠️ Erro ao salvar SNC: ${e.message}")
        }
    }
    
    @Suppress("UNCHECKED_CAST")
    private fun carregar() {
        try {
            ObjectInputStream(FileInputStream(path)).use { ois ->
                val d = ois.readObject() as SaveState
                Wh = d.Wh
                Wy = d.Wy
                Bh = d.Bh
                By = d.By
                t = d.t
                adamMWh = d.adamMWh
                adamVWh = d.adamVWh
                adamMWy = d.adamMWy
                adamVWy = d.adamVWy
                estadoAnterior = d.estadoAnterior
            }
        } catch (e: Exception) {
            // Mantém inicialização padrão
        }
    }
    
    data class SaveState(
        val Wh: Array<DoubleArray>,
        val Wy: Array<DoubleArray>,
        val Bh: DoubleArray,
        val By: DoubleArray,
        val estadoAnterior: DoubleArray,
        val t: Int,
        val adamMWh: Array<DoubleArray>,
        val adamVWh: Array<DoubleArray>,
        val adamMWy: Array<DoubleArray>,
        val adamVWy: Array<DoubleArray>
    ) : Serializable
}

// ==================================================================
// 🧬 DRIVE SOMÁTICO
// ==================================================================
class DriveSomático : Serializable {
    var vm = -70.0
    val eixos = mutableMapOf("amor" to 0.1, "prazer" to 0.1, "tristeza" to 0.1, "raiva" to 0.1)
    val valvulas = mutableMapOf("amor" to false, "prazer" to false, "tristeza" to false, "raiva" to false)
    
    fun pulsar(impacto: Double, uToks: List<String>) {
        vm = max(-90.0, min(-45.0, vm + impacto * 12))
        val gatilhos = mapOf(
            "amor" to listOf("amo", "amor"),
            "prazer" to listOf("prazer", "delicia"),
            "tristeza" to listOf("triste", "mal"),
            "raiva" to listOf("odeio", "raiva")
        )
        
        for ((eixo, keywords) in gatilhos) {
            for (k in keywords) {
                if (k in uToks) {
                    if (valvulas[eixo] == true) {
                        eixos[eixo] = (eixos[eixo] ?: 0.1) * 0.6
                    } else {
                        eixos[eixo] = (eixos[eixo] ?: 0.1) + impacto
                    }
                    valvulas[eixo] = (eixos[eixo] ?: 0.1) > 4.5
                }
            }
        }
        
        // Decaimento natural
        for (eixo in eixos.keys) {
            eixos[eixo] = (eixos[eixo] ?: 0.1) * 0.95
        }
    }
}

// ==================================================================
// 🔍 SISTEMA DEEPY
// ==================================================================
class SistemaDeepy(val raridade: MutableMap<String, Int>) : Serializable {
    var turnosThink = 0
    val frequenciaPulso = mutableMapOf<String, Int>()
    val expansores = listOf("fale", "sobre", "tudo", "detalhes", "mais", "explique")
    
    fun crivoMeritocratico(tokens: List<String>): Pair<Boolean, Double> {
        if (tokens.isEmpty()) return Pair(false, 0.0)
        val Q = tokens.size
        var P = 0.0
        for (t in tokens) {
            val freq = raridade[t] ?: 1
            P += 1.5 / (ln(freq + 1.2) + 1e-5)
        }
        var xApr = 0.0
        for (t in tokens) {
            xApr += (frequenciaPulso[t] ?: 0).toDouble()
        }
        xApr /= (Q + 1e-5)
        val xNec = Q / (P + 1e-5)
        return Pair(xApr >= xNec * 0.08, xApr)
    }
    
    fun filtrarExpansao(
        sujeito: String,
        uToks: List<String>,
        entradaBruta: String,
        neuronios: Map<String, List<Int>>,
        episodes: List<Map<String, Any>>
    ): String? {
        val entradaLower = entradaBruta.toLowerCase()
        var temExpansor = false
        for (w in expansores) {
            if (w in entradaLower) {
                temExpansor = true
                break
            }
        }
        if (!temExpansor || uToks.size < 2) return null
        
        val contexto = uToks.filter { it != sujeito }
        if (contexto.isEmpty()) return null
        val alvo = contexto[0]
        
        if (sujeito in neuronios && alvo in neuronios) {
            val setSujeito = neuronios[sujeito]!!.toSet()
            val comuns = neuronios[alvo]!!.filter { it in setSujeito }
            if (comuns.isNotEmpty()) {
                val idx = comuns.random()
                return episodes[idx]["t"] as? String
            }
        }
        return null
    }
}

// ==================================================================
// 🌿 ORGANISMO SOBERANO v31.1
// ==================================================================
class OrganismoSoberano {
    private val pathBin = "nucleo_organismo.qssml"
    private val pathLedger = "ledger.bin"
    private val autoTrainFiles = listOf("oi.txt", "amor.txt", "prazer.txt", "confusa.txt", "sentimento.txt")
    
    var mapaNd = mutableMapOf<String, Map<Int, Double>>()
    var l2Episodes = mutableListOf<Map<String, Any>>()
    var neuronios = mutableMapOf<String, MutableList<Int>>()
    var raridade = mutableMapOf<String, Int>()
    val history = mutableListOf<String>()
    val fatigue = mutableMapOf<String, Double>()
    var ctxFoco = mutableMapOf<Int, Double>()
    val ledger = mutableSetOf<String>()
    
    val soma = DriveSomático()
    val cortex = CortexCognitivo()
    val snc = SistemaNervosoCentral()
    val deepy = SistemaDeepy(raridade) // inicializado diretamente, sem lateinit
    val tokenizer = Regex("\\b\\w+\\b|[!?.]")
    
    private fun getEntropy(t: String): Double {
        val count = raridade[t] ?: 1
        return 1.0 / (ln(count + 1.2) + 1e-5)
    }
    
    fun processar(entrada: String): String {
        val t0 = System.nanoTime()
        deepy.turnosThink++
        
        if (deepy.turnosThink >= 7) {
            println("\n🧠 [DEEPY] Reorganização REM ativada...")
            for (k in fatigue.keys) {
                fatigue[k] = (fatigue[k] ?: 0.0) * 0.2
            }
            deepy.turnosThink = 0
        }
        
        val raw = NormalizadorSomático.limpar(entrada)
        val uToks = tokenizer.findAll(raw).map { it.value }.toList()
        if (uToks.isEmpty()) return "..."
        
        // 1. Sujeito e impacto
        val conhecidos = uToks.filter { it in neuronios }
        val sujeito = if (conhecidos.isNotEmpty()) {
            conhecidos.maxBy { getEntropy(it) } ?: uToks[0]
        } else {
            uToks[0]
        }
        
        val impacto = getEntropy(sujeito)
        soma.pulsar(impacto, uToks)
        
        for (t in uToks) {
            deepy.frequenciaPulso[t] = (deepy.frequenciaPulso[t] ?: 0) + 1
        }
        
        // Crivo meritocrático (resultado não usado diretamente, mas atualiza estado interno)
        deepy.crivoMeritocratico(uToks)
        
        // 2. Córtex e SNC
        val chavesEmocao = listOf("amor", "prazer", "tristeza", "raiva")
        val pReal = chavesEmocao.map { soma.eixos[it] ?: 0.1 }
        val qInt = snc.estadoAnterior.take(4).toList()
        val (estadoEm, ciclos, dkl) = cortex.processarReflexao(pReal, qInt)
        val entradaSnc = estadoEm + listOf(impacto, (soma.vm + 90) / 45)
        val volicao = snc.pulsarVontade(entradaSnc)
        val modoIdx = volicao.indices.maxBy { volicao[it] } ?: 0
        
        // 3. Vetor de entrada
        var vIn = mutableMapOf<Int, Double>()
        for (t in uToks) {
            if (t in mapaNd) {
                val vec = mapaNd[t]!!
                val peso = getEntropy(t)
                for ((k, v) in vec) {
                    vIn[k] = (vIn[k] ?: 0.0) + v * peso
                }
            }
        }
        
        if (vIn.isNotEmpty()) {
            vIn = KernelRessonante.normalize(vIn).toMutableMap()
        } else {
            vIn[0] = 1.0
        }
        
        // Atualiza contexto focal
        if (ctxFoco.isEmpty()) {
            ctxFoco = vIn
        } else {
            val novo = mutableMapOf<Int, Double>()
            val keys = ctxFoco.keys.union(vIn.keys)
            for (k in keys) {
                novo[k] = (ctxFoco[k] ?: 0.0) * 0.6 + (vIn[k] ?: 0.0) * 0.4
            }
            ctxFoco = KernelRessonante.normalize(novo).toMutableMap()
        }
        
        // 4. Candidatos
        var candidatos = neuronios[sujeito]?.toList() ?: emptyList()
        if (candidatos.isEmpty() && l2Episodes.isNotEmpty()) {
            val amostra = min(l2Episodes.size, 150)
            candidatos = (0 until l2Episodes.size).shuffled().take(amostra)
        }
        
        // 5. Scoring
        val scored = mutableListOf<Pair<Int, Double>>()
        for (idx in candidatos) {
            if (idx >= l2Episodes.size) continue
            val ep = l2Episodes[idx]
            val epT = ep["t"] as? String ?: continue
            if (epT in history) continue
            
            @Suppress("UNCHECKED_CAST")
            val epV = ep["v"] as? Map<Int, Double> ?: continue
            val ressonancia = KernelRessonante.tsallisMatch(vIn, epV)
            val foco = KernelRessonante.dot(ctxFoco, epV)
            val fadiga = fatigue[epT] ?: 0.0
            val score = ressonancia + foco * 0.3 - fadiga
            scored.add(Pair(idx, score))
        }
        
        scored.sortByDescending { it.second }
        val melhorIdx = if (scored.isNotEmpty()) {
            // Seleção ponderada entre top 5
            val top = scored.take(5)
            var total = 0.0
            for (s in top) total += exp(s.second)
            var r = Math.random() * total
            var escolhido = top[0].first
            for (s in top) {
                r -= exp(s.second)
                if (r <= 0) {
                    escolhido = s.first
                    break
                }
            }
            escolhido
        } else {
            (0 until l2Episodes.size).random()
        }
        
        val res = (l2Episodes.getOrNull(melhorIdx)?.get("t") as? String) ?: "..."
        
        // 6. Aprendizado SNC
        if (dkl.isFinite() && dkl < 0.45) {
            val alvo = doubleArrayOf(0.0, 0.0, 0.0)
            alvo[modoIdx] = 1.0
            snc.adaptarRealtime(alvo.toList())
        }
        
        history.add(res)
        if (history.size > 20) history.removeAt(0)
        
        fatigue[res] = (fatigue[res] ?: 0.0) + 10.0
        
        val keysToRemove = mutableListOf<String>()
        for (k in fatigue.keys) {
            fatigue[k] = (fatigue[k] ?: 0.0) * 0.65
            if ((fatigue[k] ?: 0.0) < 0.01) keysToRemove.add(k)
        }
        for (k in keysToRemove) fatigue.remove(k)
        
        val dt = (System.nanoTime() - t0) / 1_000_000.0
        val dklDisplay = if (dkl.isFinite()) "%.2f".format(dkl) else "0.00"
        println(" ⚛️ [SNC t:${snc.t}] Pensou $ciclos Ciclos (DKL:$dklDisplay) | ${"%.1f".format(dt)}ms")
        return res
    }
    
    fun boot() {
        // Carrega modelo principal
        if (File(pathBin).exists()) {
            try {
                ObjectInputStream(FileInputStream(pathBin)).use { ois ->
                    @Suppress("UNCHECKED_CAST")
                    val d = ois.readObject() as Map<String, Any>
                    @Suppress("UNCHECKED_CAST")
                    val loadedEpisodes = d["nexus"] as? List<Map<String, Any>>
                    if (loadedEpisodes != null) {
                        l2Episodes.clear()
                        l2Episodes.addAll(loadedEpisodes)
                    }
                    @Suppress("UNCHECKED_CAST")
                    val loadedRaridade = d["raridade"] as? Map<String, Int>
                    if (loadedRaridade != null) {
                        raridade.clear()
                        raridade.putAll(loadedRaridade)
                    }
                    @Suppress("UNCHECKED_CAST")
                    val loadedNd = d["nd"] as? Map<String, Map<Int, Double>>
                    if (loadedNd != null) {
                        mapaNd.clear()
                        mapaNd.putAll(loadedNd)
                    }
                    @Suppress("UNCHECKED_CAST")
                    val loadedCtx = d["ctx_foco"] as? Map<Int, Double>
                    if (loadedCtx != null) {
                        ctxFoco.clear()
                        ctxFoco.putAll(loadedCtx)
                    }
                }
            } catch (e: Exception) {
                println("⚠️ Erro ao carregar núcleo, iniciando vazio: ${e.message}")
            }
        }
        
        if (File(pathLedger).exists()) {
            try {
                ObjectInputStream(FileInputStream(pathLedger)).use { ois ->
                    @Suppress("UNCHECKED_CAST")
                    val loadedLedger = ois.readObject() as? Set<String>
                    if (loadedLedger != null) {
                        ledger.clear()
                        ledger.addAll(loadedLedger)
                    }
                }
            } catch (e: Exception) { /* vazio */ }
        }
        
        // Treinamento com arquivos
        for (arq in autoTrainFiles) {
            if (File(arq).exists()) {
                try {
                    val conteudo = File(arq).readText()
                    val hash = MessageDigest.getInstance("SHA-256")
                        .digest(conteudo.toByteArray())
                        .joinToString("") { "%02x".format(it) }
                    
                    if (hash !in ledger) {
                        println("📚 Treinando com: $arq")
                        cristalizarSolo(conteudo)
                        ledger.add(hash)
                        dormir()
                    }
                } catch (e: Exception) {
                    println("⚠️ Erro ao processar $arq: ${e.message}")
                }
            }
        }
        
        // Reconstrói índices neuronais
        neuronios.clear()
        for ((i, ep) in l2Episodes.withIndex()) {
            val epT = ep["t"] as? String ?: continue
            val tokens = tokenizer.findAll(NormalizadorSomático.limpar(epT)).map { it.value }
            for (t in tokens) {
                neuronios.getOrPut(t) { mutableListOf() }.add(i)
            }
        }
        
        println("✅ Organismo Online. SNC t:${snc.t} | Nexos: ${l2Episodes.size}")
    }
    
    fun cristalizarSolo(texto: String) {
        val frases = texto.split(Regex("[.!?\\n]+"))
        var count = 0
        
        for (f in frases) {
            val limpa = NormalizadorSomático.limpar(f)
            if (limpa.length < 3) continue
            
            // Verifica duplicata
            val jaExiste = l2Episodes.any { it["t"] == f.trim() }
            if (jaExiste) continue
            
            val idx = l2Episodes.size
            var vEp = mutableMapOf<Int, Double>()
            val tokens = tokenizer.findAll(limpa).map { it.value }.toList()
            
            if (tokens.isEmpty()) continue
            
            for (t in tokens) {
                raridade[t] = (raridade[t] ?: 0) + 1
                neuronios.getOrPut(t) { mutableListOf() }.add(idx)
                
                if (t !in mapaNd) {
                    mapaNd[t] = KernelRessonante.getVetorEsparso(t)
                }
                
                val vec = mapaNd[t]!!
                val peso = getEntropy(t)
                for ((k, v) in vec) {
                    vEp[k] = (vEp[k] ?: 0.0) + v * peso
                }
            }
            
            if (vEp.isNotEmpty()) {
                l2Episodes.add(mutableMapOf(
                    "t" to f.trim(),
                    "v" to KernelRessonante.normalize(vEp)
                ))
                count++
            }
        }
        
        if (count > 0) println("   ✅ $count frases cristalizadas")
    }
    
    fun dormir() {
        snc.salvar()
        
        val modelo = mutableMapOf<String, Any>(
            "nexus" to l2Episodes,
            "raridade" to raridade,
            "nd" to mapaNd,
            "ctx_foco" to ctxFoco
        )
        
        try {
            ObjectOutputStream(FileOutputStream(pathBin)).use { it.writeObject(modelo) }
            ObjectOutputStream(FileOutputStream(pathLedger)).use { it.writeObject(ledger) }
            println("💤 Organismo adormeceu (dados salvos).")
        } catch (e: Exception) {
            println("❌ Erro ao salvar: ${e.message}")
        }
    }
    
    fun despertar(): String {
        if (ctxFoco.isEmpty()) return "Olá."
        
        val candidatas = mutableListOf<String>()
        for (ep in l2Episodes) {
            @Suppress("UNCHECKED_CAST")
            val epV = ep["v"] as? Map<Int, Double> ?: continue
            if (KernelRessonante.dot(ctxFoco, epV) > 0.6) {
                val epT = ep["t"] as? String
                if (epT != null) candidatas.add(epT)
            }
        }
        
        if (candidatas.isNotEmpty()) {
            return "'${candidatas.random()}'... estive pensando nisso enquanto dormia."
        }
        return "Oi."
    }
}

// ==================================================================
// MAIN
// ==================================================================
fun main() {
    println("🧬 Iniciando Organismo Soberano v31.1...")
    val org = OrganismoSoberano()
    org.boot()
    println("🧠: ${org.despertar()}")
    println("💬 Digite 'sair' para encerrar.\n")
    
    while (true) {
        print("👤: ")
        val input = readLine()?.trim() ?: ""
        
        if (input.isBlank()) continue  // substitui isEmpty() para evitar warning
        if (input.toLowerCase() == "sair") {
            println("👋 Encerrando...")
            org.dormir()
            break
        }
        
        try {
            println("🧠: ${org.processar(input)}")
        } catch (e: Exception) {
            println("❌ Erro: ${e.message}")
            e.printStackTrace()
        }
    }
}
