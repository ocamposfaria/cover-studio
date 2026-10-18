# cover-studio

Tudo para começar um cover: baixa a música do YouTube na melhor qualidade,
separa os instrumentos localmente, de graça, com foco em **isolar a guitarra**,
e monta a pasta do cover com o projeto do GarageBand.

## Começar um cover

```bash
./init_cover.py                              # pergunta artista e música
./init_cover.py "nirvana" "lounge act"       # busca no YouTube e deixa escolher
./init_cover.py "nirvana" "lounge act" <url> # usa o link direto
```

Ele monta `~/Music/Covers/<artista> - <música>/`, baixa a música como
`<música>.mp3`, separa os stems (modo `full`), copia `templates/cover.band`
para `garageband project/<música>.band`, abre o projeto e mostra os stems no
Finder. O GarageBand não tem API para inserir faixas, então arrastar os stems
para o projeto é manual. Se o cover já existir, ele só completa o que faltar.

O template é um projeto vazio com a trilha de guitarra já montada (Brit and
Clean, Noise Gate, NeuralAmpModeler, EQ, Compressor, Tape Delay, entrada USB).
Para mudar o ponto de partida, salve outro projeto vazio por cima de
`templates/cover.band`.

## Só baixar / separar

```bash
./yt_audio.py <url>                      # só o MP3 320 kbps, em ~/Downloads
./yt_audio.py <url> --stems              # MP3 + todos os stems
./yt_audio.py <url> --stems --no-guitar  # só a base sem guitarra
./yt_audio.py                            # modo interativo
./yt_audio.py <url> --stems -c "nirvana - sappy"   # já monta a pasta do cover
```

Não precisa instalar nada: o `uv` resolve as dependências na primeira execução
(as declaradas no cabeçalho PEP 723 do script). Precisa de `ffmpeg` no PATH.

## Modos de separação

| `--stems-mode` | Sai | Tempo¹ |
|---|---|---|
| `full` (padrão) | 6 stems + guitarra dedicada + base sem guitarra (8 arquivos) | ~1m40 |
| `6s` | drums, bass, vocals, guitar, piano, other | ~40s |
| `guitar` | guitar + no_guitar | ~1m |
| `noguitar` (`--no-guitar`) | só no_guitar | ~1m |

¹ música de ~4min30, Apple Silicon com MPS (Metal).

**`no_guitar.wav` é a faixa para tocar em cima**: todos os instrumentos menos a
guitarra. Quando a guitarra vem do Roformer, essa base é feita por subtração
(`mix − guitarra`), o que **preserva sintetizador, teclado e tudo mais** em vez
de passar por um modelo que borraria esses timbres.

## Os dois modelos

**htdemucs_6s** (Meta/Demucs, MIT) — seis stems num passe. É o único jeito
gratuito de ter piano e guitarra separados. O próprio autor avisa: guitarra sai
"okay", piano vaza bastante.

**MelBand-Roformer guitar** ([becruily](https://huggingface.co/becruily/mel-band-roformer-guitar))
— dedicado só a guitarra, arquitetura mais nova, qualidade bem superior. Baixado
sob demanda (~45 MB) na primeira vez.

No modo `full` os dois rodam: o set completo vem do htdemucs_6s e a guitarra boa
sobrescreve a fraca. A do 6s fica ao lado como `guitar_6s.wav` para você comparar
de ouvido. Medido em *Sappy*: o Roformer vaza **10× menos** graves na faixa de
guitarra (0.2% vs 2.0% abaixo de 80 Hz).

## Limitação: sintetizador

Nenhum modelo aberto hoje separa sintetizador como stem próprio — o
htdemucs_6s joga synth (e órgão, e naipe de cordas) em `other`. Se o objetivo é
*remover a guitarra e preservar o synth*, use `--no-guitar`: a subtração não
toca no resto do arranjo. O que não dá hoje é o inverso, isolar o synth sozinho.

## Qualidade

Padrão é **WAV 24-bit**, o que os modelos produzem nativamente, sem re-encode.
`--stems-format flac` guarda o mesmo áudio em ~metade do espaço.

Três decisões que afetam o som, todas verificadas com medição:

- **Overlap-add no Roformer.** Ele processa em janelas de 11 s. Emendar janelas
  direto deixa um clique audível a cada emenda (salto 2.6× maior que o
  transiente mais alto da faixa). Com janela de Hann e hop de 50% isso cai para
  0.9×, ou seja, abaixo do conteúdo normal. Custa ~57% mais tempo.
- **Resample obrigatório.** O YouTube entrega Opus a 48 kHz e o Roformer foi
  treinado em 44.1 kHz. Sem resample o modelo nem roda — e era o motivo de a
  guitarra boa cair silenciosamente no fallback. Usa soxr quando o ffmpeg tem,
  senão o resampler nativo com filtro longo.
- **Headroom.** Somar ou subtrair stems de música masterizada estoura 0 dBFS.
  Tudo é reescalado para deixar ~0.3 dB de folga.

`--shifts N` faz N passes do demucs com deslocamento aleatório: melhora um
pouco, custa N× o tempo. `--device cpu` se o MPS der problema (é ~1.8× mais
lento).

## Estrutura

Um cover tem esta cara (o `init_cover.py` e o `--cover` montam igual):

```
~/Music/Covers/nirvana - sappy/
├── song and stems/       a música e os stems (o script escreve aqui)
│   ├── sappy.mp3
│   └── stems/
│       ├── guitar.wav       # Roformer dedicado
│       ├── guitar_6s.wav    # htdemucs_6s, para comparar
│       ├── no_guitar.wav    # ← tocar em cima
│       └── bass.wav  drums.wav  vocals.wav  piano.wav  other.wav
├── garageband project/   o .band (copiado do template)
├── exported mixes/       os bounces de áudio
└── final video/          o vídeo do cover
```

Os perfis de amp ficam à parte, numa biblioteca por artista compartilhada entre
covers — `~/Music/NAM/Kurt Cobain/`, `~/Music/NAM/ZZ Top/` etc.

## Nota sobre licença

O Demucs é MIT. O checkpoint de guitarra do becruily **não tem licença
declarada** no repositório do HuggingFace — as menções a MIT que circulam vêm
de mirrors, não da fonte. Para uso pessoal é o que existe de melhor; se for usar
comercialmente, confirme antes.
