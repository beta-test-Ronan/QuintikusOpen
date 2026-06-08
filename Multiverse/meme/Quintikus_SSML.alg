Algoritmo "OrganismoSoberano_v31_Visualg"
// ======================================================
// 🧬 Organismo Soberano v31.1 – Visualg (Portugol)
// ======================================================

Var
   // Cérebro
   episodios: vetor[1..1000] de caractere
   totalEpisodios: inteiro
   // Histórico e fadiga
   historico: vetor[1..20] de caractere
   historicoIdx: inteiro
   fadiga: vetor[1..1000] de real
   // Emoções
   vm, amor, prazer, tristeza, raiva: real
   // Auxiliares
   entrada, resposta, sujeito: caractere
   i, j, melhor, ciclos: inteiro
   score, melhorScore: real

// -------------------------------------------------------
// Função Normalizar: tira acentos e minúsculas
// -------------------------------------------------------
Funcao Normalizar(texto: caractere): caractere
Var
   resultado: caractere
   c: caractere
   pos: inteiro
Inicio
   resultado <- ""
   texto <- Minusc(texto)
   Para pos <- 1 ate Compr(texto) faca
      c <- Copia(texto, pos, 1)
      // Remove acentos simples
      Escolha c
         Caso "á", "à", "â", "ã": c <- "a"
         Caso "é", "è", "ê": c <- "e"
         Caso "í", "ì", "î": c <- "i"
         Caso "ó", "ò", "ô", "õ": c <- "o"
         Caso "ú", "ù", "û": c <- "u"
         Caso "ç": c <- "c"
      FimEscolha
      Se (c >= "a" e c <= "z") ou (c >= "0" e c <= "9") ou (c = "!") ou (c = "?") ou (c = ".") ou (c = " ") entao
         resultado <- resultado + c
      FimSe
   FimPara
   Enquanto (Copia(resultado, 1, 1) = " ") faca
      resultado <- Copia(resultado, 2, Compr(resultado)-1)
   FimEnquanto
   Enquanto (Copia(resultado, Compr(resultado), 1) = " ") faca
      resultado <- Copia(resultado, 1, Compr(resultado)-1)
   FimEnquanto
   Retorne resultado
FimFuncao

// -------------------------------------------------------
// Função Entropia: quanto mais rara a palavra, maior o valor
// -------------------------------------------------------
Funcao Entropia(palavra: caractere): real
Var
   contagem, i: inteiro
Inicio
   contagem <- 0
   Para i <- 1 ate totalEpisodios faca
      Se (Pos(palavra, Minusc(episodios[i])) > 0) entao
         contagem <- contagem + 1
      FimSe
   FimPara
   Se contagem = 0 entao
      contagem <- 1
   FimSe
   Retorne 1.0 / (Log(contagem + 1.2) + 0.00001)
FimFuncao

// -------------------------------------------------------
// Função Similaridade: calcula semelhança entre duas strings
// -------------------------------------------------------
Funcao Similaridade(frase1, frase2: caractere): real
Var
   i, j, acertos, total: inteiro
   palavra1, palavra2: vetor[1..50] de caractere
   num1, num2: inteiro
   encontrou: logico
Inicio
   // Quebrar frases em palavras
   num1 <- 0
   palavra1[1] <- ""
   Para i <- 1 ate Compr(frase1) faca
      Se Copia(frase1, i, 1) = " " entao
         Se Compr(palavra1[num1+1]) > 0 entao
            num1 <- num1 + 1
            palavra1[num1+1] <- ""
         FimSe
      Senao
         palavra1[num1+1] <- palavra1[num1+1] + Copia(frase1, i, 1)
      FimSe
   FimPara
   Se Compr(palavra1[num1+1]) > 0 entao
      num1 <- num1 + 1
   FimSe

   num2 <- 0
   palavra2[1] <- ""
   Para i <- 1 ate Compr(frase2) faca
      Se Copia(frase2, i, 1) = " " entao
         Se Compr(palavra2[num2+1]) > 0 entao
            num2 <- num2 + 1
            palavra2[num2+1] <- ""
         FimSe
      Senao
         palavra2[num2+1] <- palavra2[num2+1] + Copia(frase2, i, 1)
      FimSe
   FimPara
   Se Compr(palavra2[num2+1]) > 0 entao
      num2 <- num2 + 1
   FimSe

   acertos <- 0
   Para i <- 1 ate num1 faca
      encontrou <- Falso
      Para j <- 1 ate num2 faca
         Se palavra1[i] = palavra2[j] entao
            encontrou <- Verdadeiro
         FimSe
      FimPara
      Se encontrou entao
         acertos <- acertos + 1
      FimSe
   FimPara

   total <- num1 + num2
   Se total = 0 entao
      Retorne 0
   FimSe
   Retorne (2.0 * acertos) / total
FimFuncao

// -------------------------------------------------------
// Função Processar: coração da Lucy
// -------------------------------------------------------
Funcao Processar(entrada: caractere): caractere
Var
   sujeito, palavra: caractere
   impacto, dkl: real
   i, melhorIdx: inteiro
   melhorScore, score, fadigaAtual: real
   inicio, fim: inteiro
   token: vetor[1..50] de caractere
   numTokens: inteiro
Inicio
   // Normalizar entrada
   entrada <- Normalizar(entrada)
   Se Compr(entrada) = 0 entao
      Retorne "..."
   FimSe

   // Extrair tokens
   numTokens <- 0
   token[1] <- ""
   Para i <- 1 ate Compr(entrada) faca
      Se Copia(entrada, i, 1) = " " entao
         Se Compr(token[numTokens+1]) > 0 entao
            numTokens <- numTokens + 1
            token[numTokens+1] <- ""
         FimSe
      Senao
         token[numTokens+1] <- token[numTokens+1] + Copia(entrada, i, 1)
      FimSe
   FimPara
   Se Compr(token[numTokens+1]) > 0 entao
      numTokens <- numTokens + 1
   FimSe

   // Identificar sujeito principal
   sujeito <- token[1]
   impacto <- Entropia(sujeito)

   // Pulsar drive somático
   vm <- vm + impacto * 12
   Se vm > -45 entao vm <- -45 FimSe
   Se vm < -90 entao vm <- -90 FimSe

   Se Pos("amor", entrada) > 0 ou Pos("amo", entrada) > 0 entao
      amor <- amor + impacto
   FimSe
   Se Pos("triste", entrada) > 0 entao
      tristeza <- tristeza + impacto
   FimSe

   // Buscar melhor episódio
   melhorIdx <- 0
   melhorScore <- -999999.0
   Para i <- 1 ate totalEpisodios faca
      // Pular se está no histórico recente
      Se (historico[historicoIdx] = episodios[i]) ou
         (historico[(historicoIdx-1+20)%20+1] = episodios[i]) entao
         // Não usa
      Senao
         score <- Similaridade(entrada, Minusc(episodios[i]))
         fadigaAtual <- fadiga[i]
         score <- score - fadigaAtual
         Se score > melhorScore entao
            melhorScore <- score
            melhorIdx <- i
         FimSe
      FimSe
   FimPara

   Se melhorIdx = 0 entao
      melhorIdx <- 1
   FimSe

   resposta <- episodios[melhorIdx]

   // Atualizar histórico e fadiga
   historicoIdx <- (historicoIdx % 20) + 1
   historico[historicoIdx] <- resposta
   fadiga[melhorIdx] <- fadiga[melhorIdx] + 10.0

   // Decair fadiga geral
   Para i <- 1 ate totalEpisodios faca
      fadiga[i] <- fadiga[i] * 0.65
   FimPara

   Retorne resposta
FimFuncao

// -------------------------------------------------------
// Função Treinar: carrega frases de um vetor
// -------------------------------------------------------
Procedimento Treinar(frases: vetor[1..100] de caractere; num: inteiro)
Var
   i: inteiro
Inicio
   Para i <- 1 ate num faca
      totalEpisodios <- totalEpisodios + 1
      episodios[totalEpisodios] <- frases[i]
   FimPara
FimProcedimento

// -------------------------------------------------------
// PROGRAMA PRINCIPAL
// -------------------------------------------------------
Inicio
   // Inicializar
   totalEpisodios <- 0
   vm <- -70.0
   amor <- 0.1
   prazer <- 0.1
   tristeza <- 0.1
   raiva <- 0.1
   Para i <- 1 ate 20 faca
      historico[i] <- ""
   FimPara
   historicoIdx <- 0

   // Treinar com algumas frases
   totalEpisodios <- 0
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "Oi, amor"
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "E eu tambem amo o seu jeito, sabia"
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "Quando pergunta como estou, eu me sinto bem"
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "Nossa conversa e como musica"
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "Estou aqui para voce"
   totalEpisodios <- totalEpisodios + 1
   episodios[totalEpisodios] <- "Meu coracao acalma quando recebo seu oi"

   EscrevaL("🧬 Organismo Soberano v31.1 – Visualg")
   EscrevaL("Lucy: Olá! Digite 'sair' para encerrar.")

   Repita
      Escreva("👤: ")
      Leia(entrada)
      Se Minusc(entrada) = "sair" entao
         Interrompa
      FimSe
      resposta <- Processar(entrada)
      EscrevaL("🧠 Lucy: ", resposta)
   Ate Falso

   EscrevaL("💤 Lucy adormeceu. Até mais!")
FimAlgoritmo