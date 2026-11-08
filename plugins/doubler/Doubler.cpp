// Doubler: duplica a guitarra e abre em estéreo (efeito Haas).
//
// Esquerda = o sinal como chegou, sem latência nenhuma (dá para tocar
// monitorando ao vivo). Direita = o mesmo sinal atrasado pelo parâmetro
// "Delay". Com 10-30 ms o ouvido funde os dois lados numa guitarra larga,
// como se fossem duas gravações; acima de ~40 ms começa a soar como eco.
//
// Entra mono (1→2) ou estéreo (2→2). No estéreo a esquerda passa intacta e a
// direita sai atrasada.

#include <AudioUnitSDK/AUEffectBase.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr AudioUnitParameterID kParamDelay = 0;
constexpr float kDelayMinMs = 0.0f;
constexpr float kDelayMaxMs = 100.0f;
constexpr float kDelayDefaultMs = 20.0f;

// Mexer no delay com o som rolando faz o tempo deslizar até o valor novo em
// vez de pular, o que daria um clique. 50 ms de constante de tempo.
constexpr double kGlideSeconds = 0.05;

} // namespace

class Doubler final : public ausdk::AUEffectBase {
public:
	explicit Doubler(AudioComponentInstance ci) : AUEffectBase(ci, /*inProcessesInPlace=*/false)
	{
		CreateElements();
		Globals()->UseIndexedParameters(1);
		SetParameter(kParamDelay, kDelayDefaultMs);
	}

	OSStatus Initialize() override
	{
		AUSDK_Require_noerr(AUEffectBase::Initialize());
		const double sampleRate = GetSampleRate();
		// +4: folga para a interpolação nunca ler a amostra que acabou de ser escrita.
		mLine.assign(static_cast<size_t>(std::ceil(sampleRate * kDelayMaxMs / 1000.0)) + 4, 0.0f);
		mWrite = 0;
		mGlideCoef = 1.0 - std::exp(-1.0 / (kGlideSeconds * sampleRate));
		mDelaySamples = TargetDelaySamples();
		return noErr;
	}

	void Cleanup() override
	{
		AUEffectBase::Cleanup();
		mLine.clear();
		mLine.shrink_to_fit();
	}

	OSStatus Reset(AudioUnitScope scope, AudioUnitElement element) override
	{
		std::fill(mLine.begin(), mLine.end(), 0.0f);
		mWrite = 0;
		mDelaySamples = TargetDelaySamples();
		return AUEffectBase::Reset(scope, element);
	}

	UInt32 SupportedNumChannels(const AUChannelInfo** outInfo) override
	{
		static constexpr AUChannelInfo kConfigs[] = { { 1, 2 }, { 2, 2 } };
		if (outInfo != nullptr) {
			*outInfo = kConfigs;
		}
		return 2;
	}

	OSStatus GetParameterInfo(AudioUnitScope scope, AudioUnitParameterID id,
		AudioUnitParameterInfo& info) override
	{
		AUSDK_Require(scope == kAudioUnitScope_Global, kAudioUnitErr_InvalidScope);
		AUSDK_Require(id == kParamDelay, kAudioUnitErr_InvalidParameter);
		info.flags = kAudioUnitParameterFlag_IsReadable | kAudioUnitParameterFlag_IsWritable;
		FillInParameterName(info, CFSTR("Delay"), false);
		info.unit = kAudioUnitParameterUnit_Milliseconds;
		info.minValue = kDelayMinMs;
		info.maxValue = kDelayMaxMs;
		info.defaultValue = kDelayDefaultMs;
		return noErr;
	}

	bool SupportsTail() noexcept AUSDK_RTSAFE override { return true; }
	Float64 GetTailTime() noexcept AUSDK_RTSAFE override { return kDelayMaxMs / 1000.0; }

	OSStatus ProcessBufferLists(AudioUnitRenderActionFlags& ioActionFlags,
		const AudioBufferList& in, AudioBufferList& out, UInt32 frames) noexcept AUSDK_RTSAFE override
	{
		AUSDK_Require(!mLine.empty(), kAudioUnitErr_Uninitialized);
		AUSDK_Require(in.mNumberBuffers >= 1 && out.mNumberBuffers >= 2,
			kAudioUnitErr_FormatNotSupported);

		// Só declara silêncio depois que a entrada ficou muda por mais tempo
		// que o delay máximo; até lá a direita ainda tem rabo para tocar.
		if (IsInputSilent(ioActionFlags, frames)) {
			ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
			return noErr;
		}
		ioActionFlags &= ~kAudioUnitRenderAction_OutputIsSilence;

		const auto* srcL = static_cast<const float*>(in.mBuffers[0].mData);
		const auto* srcR =
			static_cast<const float*>(in.mBuffers[in.mNumberBuffers > 1 ? 1 : 0].mData);
		auto* dstL = static_cast<float*>(out.mBuffers[0].mData);
		auto* dstR = static_cast<float*>(out.mBuffers[1].mData);

		const size_t size = mLine.size();
		const double target = TargetDelaySamples();
		double delay = mDelaySamples;

		for (UInt32 i = 0; i < frames; ++i) {
			// Lê as duas entradas antes de escrever qualquer saída: o host pode
			// passar o mesmo buffer para entrada e saída.
			const float left = srcL[i];
			const float right = srcR[i];

			mLine[mWrite] = right;
			delay += (target - delay) * mGlideCoef;

			double readPos = static_cast<double>(mWrite) - delay;
			if (readPos < 0.0) {
				readPos += static_cast<double>(size);
			}
			auto i0 = static_cast<size_t>(readPos);
			const auto frac = static_cast<float>(readPos - static_cast<double>(i0));
			if (i0 >= size) {
				i0 -= size;
			}
			const size_t i1 = (i0 + 1 == size) ? 0 : i0 + 1;

			dstL[i] = left;
			dstR[i] = mLine[i0] + (mLine[i1] - mLine[i0]) * frac;

			if (++mWrite == size) {
				mWrite = 0;
			}
		}

		mDelaySamples = delay;
		return noErr;
	}

private:
	double TargetDelaySamples() noexcept AUSDK_RTSAFE
	{
		const float ms = std::clamp(GetParameter(kParamDelay), kDelayMinMs, kDelayMaxMs);
		return static_cast<double>(ms) * GetSampleRate() / 1000.0;
	}

	std::vector<float> mLine;
	size_t mWrite = 0;
	double mDelaySamples = 0.0;
	double mGlideCoef = 1.0;
};

AUSDK_COMPONENT_ENTRY(ausdk::AUBaseFactory, Doubler)
