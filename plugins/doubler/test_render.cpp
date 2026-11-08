// Teste do Doubler: carrega o binário compilado (antes de instalar), registra
// no processo e confere amostra a amostra o que sai de cada lado.
//
//   test_render build/Doubler.component/Contents/MacOS/Doubler

#include <AudioToolbox/AudioToolbox.h>
#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

int gFailures = 0;

#define CHECK(cond, ...)                                                                           \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::printf("    ✗ ");                                                                 \
			std::printf(__VA_ARGS__);                                                              \
			std::printf("\n");                                                                     \
			++gFailures;                                                                           \
		}                                                                                          \
	} while (0)

void Must(OSStatus err, const char* what)
{
	if (err != noErr) {
		std::printf("    ✗ %s → OSStatus %d\n", what, static_cast<int>(err));
		std::exit(1);
	}
}

using Channels = std::vector<std::vector<float>>;

struct Source {
	Channels ch;
	size_t pos = 0;
};

OSStatus Feed(void* ref, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 frames,
	AudioBufferList* io)
{
	auto* src = static_cast<Source*>(ref);
	for (UInt32 b = 0; b < io->mNumberBuffers; ++b) {
		auto* dst = static_cast<float*>(io->mBuffers[b].mData);
		const auto& c = src->ch[b];
		for (UInt32 i = 0; i < frames; ++i) {
			const size_t p = src->pos + i;
			dst[i] = p < c.size() ? c[p] : 0.0f;
		}
	}
	src->pos += frames;
	return noErr;
}

AudioStreamBasicDescription Format(double sr, UInt32 channels)
{
	AudioStreamBasicDescription f{};
	f.mSampleRate = sr;
	f.mFormatID = kAudioFormatLinearPCM;
	f.mFormatFlags = static_cast<AudioFormatFlags>(kAudioFormatFlagsNativeFloatPacked) | kAudioFormatFlagIsNonInterleaved;
	f.mBytesPerPacket = f.mBytesPerFrame = sizeof(float);
	f.mFramesPerPacket = 1;
	f.mChannelsPerFrame = channels;
	f.mBitsPerChannel = 32;
	return f;
}

struct StereoBufferList {
	UInt32 mNumberBuffers = 2;
	AudioBuffer mBuffers[2];
};

class Rig {
public:
	Rig(AudioComponent comp, double sr, UInt32 inChannels, float delayMs) : mSampleRate(sr)
	{
		Must(AudioComponentInstanceNew(comp, &mUnit), "instanciar");
		const auto inFmt = Format(sr, inChannels);
		const auto outFmt = Format(sr, 2);
		Must(AudioUnitSetProperty(mUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
				 &inFmt, sizeof(inFmt)),
			"formato de entrada");
		Must(AudioUnitSetProperty(mUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0,
				 &outFmt, sizeof(outFmt)),
			"formato de saída");
		const UInt32 maxFrames = 1024;
		Must(AudioUnitSetProperty(mUnit, kAudioUnitProperty_MaximumFramesPerSlice,
				 kAudioUnitScope_Global, 0, &maxFrames, sizeof(maxFrames)),
			"max frames");
		AURenderCallbackStruct cb{ Feed, &mSource };
		Must(AudioUnitSetProperty(mUnit, kAudioUnitProperty_SetRenderCallback,
				 kAudioUnitScope_Input, 0, &cb, sizeof(cb)),
			"render callback");
		SetDelay(delayMs);
		mInitStatus = AudioUnitInitialize(mUnit);
	}

	~Rig()
	{
		AudioUnitUninitialize(mUnit);
		AudioComponentInstanceDispose(mUnit);
	}

	OSStatus InitStatus() const { return mInitStatus; }
	AudioUnit Unit() const { return mUnit; }

	void SetDelay(float ms)
	{
		Must(AudioUnitSetParameter(mUnit, 0, kAudioUnitScope_Global, 0, ms, 0), "set delay");
	}

	void SetInput(Channels ch)
	{
		mSource.ch = std::move(ch);
		mSource.pos = 0;
	}

	Channels Render(size_t total, UInt32 block = 256)
	{
		Channels out(2, std::vector<float>(total, 0.0f));
		std::vector<float> l(block), r(block);
		for (size_t done = 0; done < total; done += block) {
			const auto n = static_cast<UInt32>(std::min<size_t>(block, total - done));
			StereoBufferList abl;
			abl.mBuffers[0] = { 1, n * 4, l.data() };
			abl.mBuffers[1] = { 1, n * 4, r.data() };
			AudioTimeStamp ts{};
			ts.mSampleTime = mSampleTime;
			ts.mFlags = kAudioTimeStampSampleTimeValid;
			AudioUnitRenderActionFlags flags = 0;
			Must(AudioUnitRender(
					 mUnit, &flags, &ts, 0, n, reinterpret_cast<AudioBufferList*>(&abl)),
				"render");
			const bool silent = (flags & kAudioUnitRenderAction_OutputIsSilence) != 0;
			for (UInt32 i = 0; i < n; ++i) {
				out[0][done + i] = silent ? 0.0f : static_cast<float*>(abl.mBuffers[0].mData)[i];
				out[1][done + i] = silent ? 0.0f : static_cast<float*>(abl.mBuffers[1].mData)[i];
			}
			mSampleTime += n;
		}
		return out;
	}

private:
	AudioUnit mUnit = nullptr;
	Source mSource;
	double mSampleRate;
	double mSampleTime = 0;
	OSStatus mInitStatus = noErr;
};

std::vector<float> Impulse(size_t len, size_t at)
{
	std::vector<float> v(len, 0.0f);
	v[at] = 1.0f;
	return v;
}

// Soma e centro de massa de um trecho: localiza o impulso mesmo quando ele cai
// entre duas amostras (delay fracionário).
void Moments(const std::vector<float>& v, double& sum, double& center)
{
	sum = 0;
	double weighted = 0;
	for (size_t i = 0; i < v.size(); ++i) {
		sum += v[i];
		weighted += v[i] * static_cast<double>(i);
	}
	center = sum != 0 ? weighted / sum : -1;
}

double MaxAbs(const std::vector<float>& v, size_t from, size_t to)
{
	double m = 0;
	for (size_t i = from; i < to && i < v.size(); ++i) {
		m = std::max(m, static_cast<double>(std::fabs(v[i])));
	}
	return m;
}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::printf("uso: %s <binário do Doubler>\n", argv[0]);
		return 2;
	}
	void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (lib == nullptr) {
		std::printf("dlopen: %s\n", dlerror());
		return 1;
	}
	auto factory = reinterpret_cast<AudioComponentFactoryFunction>(dlsym(lib, "DoublerFactory"));
	if (factory == nullptr) {
		std::printf("DoublerFactory não exportada\n");
		return 1;
	}
	// Subtype próprio do teste para não colidir com uma versão já instalada.
	AudioComponentDescription desc{ kAudioUnitType_Effect, 'Dbts', 'CvSt', 0, 0 };
	AudioComponent comp =
		AudioComponentRegister(&desc, CFSTR("Cover Studio: Doubler (teste)"), 0x10000, factory);

	{
		std::printf("  mono → estéreo, 48 kHz, 20 ms (padrão)\n");
		Rig rig(comp, 48000, 1, 20.0f);
		Must(rig.InitStatus(), "inicializar 1→2");
		rig.SetInput({ Impulse(4800, 100) });
		auto out = rig.Render(4800);
		CHECK(out[0][100] == 1.0f && MaxAbs(out[0], 101, 4800) == 0 && MaxAbs(out[0], 0, 100) == 0,
			"esquerda deveria ser o impulso intacto em 100");
		CHECK(out[1][100 + 960] == 1.0f, "direita deveria ter o impulso em 1060, tem %g",
			out[1][1060]);
		CHECK(MaxAbs(out[1], 0, 1060) == 0 && MaxAbs(out[1], 1061, 4800) == 0,
			"direita com sobra fora de 1060");
	}

	{
		std::printf("  estéreo → estéreo, 44.1 kHz, 7.5 ms (delay fracionário)\n");
		Rig rig(comp, 44100, 2, 7.5f);
		Must(rig.InitStatus(), "inicializar 2→2");
		// Sinais diferentes em cada lado para pegar vazamento entre canais.
		rig.SetInput({ Impulse(4410, 50), Impulse(4410, 200) });
		auto out = rig.Render(4410);
		double sumL, cL, sumR, cR;
		Moments(out[0], sumL, cL);
		Moments(out[1], sumR, cR);
		CHECK(out[0][50] == 1.0f && std::fabs(sumL - 1) < 1e-6, "esquerda deveria passar intacta");
		CHECK(std::fabs(sumR - 1) < 1e-5, "direita deveria ter ganho 1, tem %g", sumR);
		CHECK(std::fabs(cR - (200 + 330.75)) < 1e-3, "direita centrada em 530.75, está em %g", cR);
	}

	{
		std::printf("  girar o delay com som rolando: desliza sem clique\n");
		const double sr = 48000;
		Rig rig(comp, sr, 1, 20.0f);
		Must(rig.InitStatus(), "inicializar");
		const size_t n = 48000 * 2;
		std::vector<float> sine(n);
		for (size_t i = 0; i < n; ++i) {
			sine[i] = static_cast<float>(std::sin(2 * std::numbers::pi * 440 * i / sr));
		}
		rig.SetInput({ sine });
		auto a = rig.Render(24000);
		rig.SetDelay(40.0f);
		auto b = rig.Render(n - 24000);
		std::vector<float> right = a[1];
		right.insert(right.end(), b[1].begin(), b[1].end());
		double maxStep = 0;
		for (size_t i = 24000; i < 24000 + 12000; ++i) {
			maxStep = std::max(maxStep, static_cast<double>(std::fabs(right[i] - right[i - 1])));
		}
		// Seno de 440 Hz anda no máximo 0.058 por amostra; um pulo de delay daria até 2.
		CHECK(maxStep < 0.12, "salto de %g entre amostras durante a troca", maxStep);
		double err = 0;
		for (size_t i = n - 4800; i < n; ++i) {
			err = std::max(err, static_cast<double>(std::fabs(right[i] - sine[i - 1920])));
		}
		CHECK(err < 1e-3, "depois de assentar, direita deveria ser a entrada 40 ms atrás (erro %g)",
			err);
	}

	{
		std::printf("  bypass: os dois lados recebem o sinal seco\n");
		Rig rig(comp, 48000, 1, 20.0f);
		Must(rig.InitStatus(), "inicializar");
		const UInt32 on = 1;
		Must(AudioUnitSetProperty(rig.Unit(), kAudioUnitProperty_BypassEffect,
				 kAudioUnitScope_Global, 0, &on, sizeof(on)),
			"bypass");
		rig.SetInput({ Impulse(2048, 10) });
		auto out = rig.Render(2048);
		CHECK(out[0][10] == 1.0f && out[1][10] == 1.0f && MaxAbs(out[1], 11, 2048) == 0,
			"bypass deveria repetir a entrada nos dois lados");
	}

	{
		std::printf("  estado salvo no projeto volta igual\n");
		Rig rig(comp, 48000, 1, 33.0f);
		Must(rig.InitStatus(), "inicializar");
		CFPropertyListRef state = nullptr;
		UInt32 size = sizeof(state);
		Must(AudioUnitGetProperty(rig.Unit(), kAudioUnitProperty_ClassInfo, kAudioUnitScope_Global,
				 0, &state, &size),
			"ler ClassInfo");
		rig.SetDelay(5.0f);
		Must(AudioUnitSetProperty(rig.Unit(), kAudioUnitProperty_ClassInfo, kAudioUnitScope_Global,
				 0, &state, sizeof(state)),
			"restaurar ClassInfo");
		CFRelease(state);
		AudioUnitParameterValue v = 0;
		Must(AudioUnitGetParameter(rig.Unit(), 0, kAudioUnitScope_Global, 0, &v), "ler delay");
		CHECK(v == 33.0f, "delay restaurado deveria ser 33, é %g", v);
	}

	{
		std::printf("  mono → mono é recusado (sem estéreo não há duplicação)\n");
		AudioUnit unit = nullptr;
		Must(AudioComponentInstanceNew(comp, &unit), "instanciar");
		const auto mono = Format(48000, 1);
		AudioUnitSetProperty(
			unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &mono, sizeof(mono));
		AudioUnitSetProperty(
			unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &mono, sizeof(mono));
		CHECK(AudioUnitInitialize(unit) == kAudioUnitErr_FormatNotSupported,
			"1→1 deveria falhar com FormatNotSupported");
		AudioComponentInstanceDispose(unit);
	}

	if (gFailures == 0) {
		std::printf("✓ todos os testes passaram\n");
		return 0;
	}
	std::printf("✗ %d falha(s)\n", gFailures);
	return 1;
}
