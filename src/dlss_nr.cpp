#include "dlss_nr.hpp"

#include "d3d12_output_contract.hpp"
#include "dlss_nr_contract.hpp"
#include "gaze_foveation.hpp"
#include "runtime.hpp"

#include <Windows.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <deque>
#include <mutex>
#include <utility>

namespace cheeky::foveated_dlss {
namespace {

using NgxInitExtFn = NgxResult (*)(
    unsigned long long,
    const wchar_t*,
    ID3D12Device*,
    std::uint32_t,
    const NgxParameters*
);
using NgxAllocateParametersFn = NgxResult (*)(NgxParameters**);
using NgxDestroyParametersFn = NgxResult (*)(NgxParameters*);
using NgxCreateFeatureFn = NgxResult (*)(
    ID3D12GraphicsCommandList*,
    std::uint32_t,
    NgxParameters*,
    NgxHandle**
);
using NgxEvaluateFeatureFn = NgxResult (*)(
    ID3D12GraphicsCommandList*,
    const NgxHandle*,
    const NgxParameters*,
    NgxProgressCallback
);
using NgxReleaseFeatureFn = NgxResult (*)(NgxHandle*);
using GetModuleFileNameWFn = DWORD (WINAPI*)(HMODULE, LPWSTR, DWORD);

template <typename T>
void release(T*& object) noexcept {
    if (object != nullptr) {
        object->Release();
        object = nullptr;
    }
}

[[nodiscard]] bool same_d3d12_device(
    ID3D12Device* const left,
    ID3D12Device* const right
) noexcept {
    if (left == right) return true;
    if (left == nullptr || right == nullptr) return false;
    IUnknown* left_identity{};
    IUnknown* right_identity{};
    const bool left_ok = SUCCEEDED(left->QueryInterface(IID_PPV_ARGS(&left_identity)));
    const bool right_ok = SUCCEEDED(right->QueryInterface(IID_PPV_ARGS(&right_identity)));
    const bool same = left_ok && right_ok && left_identity == right_identity;
    release(left_identity);
    release(right_identity);
    return same;
}

struct RuntimeState {
    HMODULE module{};
    HMODULE addon{};
    ID3D12Device* device{};
    NgxAllocateParametersFn allocate_parameters{};
    NgxDestroyParametersFn destroy_parameters{};
    NgxCreateFeatureFn create_feature{};
    NgxEvaluateFeatureFn evaluate_feature{};
    NgxReleaseFeatureFn release_feature{};
    GetModuleFileNameWFn get_module_file_name{};
    // 0 not attempted, 1 ready, 2 missing DLL, 3 failed.
    std::uint32_t state{};
};

std::atomic<bool> host_evaluate_succeeded{};
std::atomic<bool> runtime_load_started{};
ID3D12Device* pending_device{};

// Feature 18 sees a 0-1 Reinhard proxy, not Luma's linear HDR Color.
// Bit 0 (IsHDR) would tell NR the input is linear HDR; that clipped
// highlights to 1 and crushed green. Keep MV-low-res and depth-inverted.
[[nodiscard]] constexpr std::uint32_t nr_feature_create_flags(
    const std::uint32_t game_create_flags
) noexcept {
    return game_create_flags & ~1U;
}

struct FeatureKey {
    std::uint32_t input_width{};
    std::uint32_t input_height{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
    std::uint32_t create_flags{};
};

struct NrRegion {
    std::uint32_t base_x{};
    std::uint32_t base_y{};
    std::uint32_t width{};
    std::uint32_t height{};
    float shape_width{1.0F};
    float shape_height{1.0F};
    float roundness{};
    float transition{};
};

[[nodiscard]] bool operator==(
    const FeatureKey& left,
    const FeatureKey& right
) noexcept {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

struct GpuResources {
    ID3D12Resource* game_output{};
    ID3D12Resource* original_output{};
    ID3D12Resource* color_proxy{};
    ID3D12Resource* neural_output{};
    ID3D12DescriptorHeap* descriptors{};
    ID3D12RootSignature* root_signature{};
    ID3D12PipelineState* encode_pipeline{};
    ID3D12PipelineState* decode_pipeline{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t working_width{};
    std::uint32_t working_height{};
    std::uint32_t descriptor_size{};
};

struct CachedFeature {
    FeatureKey key{};
    NgxParameters* parameters{};
    NgxHandle* handle{};
};

struct ViewState {
    DlssViewId view_id{};
    NgxParameters* parameters{};
    NgxHandle* handle{};
    FeatureKey key{};
    bool has_key{};
    bool feature_failed{};
    bool was_enabled{};
    std::uint64_t settings_signature{};
    std::uint64_t reset_generation{};
    std::uint32_t logged_crop_x{};
    std::uint32_t logged_crop_y{};
    std::uint32_t logged_crop_w{};
    std::uint32_t logged_crop_h{};
    bool logged_crop{};
    std::deque<CachedFeature> retired_features;
    std::deque<GpuResources> gpu_resources;
    ID3D12Resource* input_scratch{};
    std::uint32_t scratch_width{};
    std::uint32_t scratch_height{};
    DXGI_FORMAT scratch_format{DXGI_FORMAT_UNKNOWN};
    ID3D12Resource* sample_readback{};
    bool samples_pending{};
    std::uint32_t sample_crop_width{};
    std::uint32_t sample_crop_height{};
    std::uint32_t sample_proxy_width{};
    std::uint32_t sample_proxy_height{};
    std::uint32_t sample_format{};
    float sample_paper_white{1.0F};
};

std::mutex nr_mutex;
RuntimeState runtime;
std::deque<ViewState> views;
std::atomic<std::uint64_t> requested_reset_generation{};
DlssNrSnapshot diagnostics;
constexpr std::size_t retired_feature_capacity = 8U;
constexpr std::size_t gpu_resource_cache_capacity = 8U;

void release_gpu(GpuResources& gpu) noexcept {
    release(gpu.decode_pipeline);
    release(gpu.encode_pipeline);
    release(gpu.root_signature);
    release(gpu.descriptors);
    release(gpu.neural_output);
    release(gpu.color_proxy);
    release(gpu.original_output);
    release(gpu.game_output);
    gpu = {};
}

void release_feature(ViewState& view) noexcept {
    if (view.handle != nullptr && runtime.release_feature != nullptr) {
        static_cast<void>(runtime.release_feature(view.handle));
    }
    if (view.parameters != nullptr && runtime.destroy_parameters != nullptr) {
        static_cast<void>(runtime.destroy_parameters(view.parameters));
    }
    view.handle = nullptr;
    view.parameters = nullptr;
    view.has_key = false;
    view.feature_failed = false;
    for (const auto& retired : view.retired_features) {
        if (retired.handle != nullptr && runtime.release_feature != nullptr) {
            static_cast<void>(runtime.release_feature(retired.handle));
        }
        if (retired.parameters != nullptr && runtime.destroy_parameters != nullptr) {
            static_cast<void>(runtime.destroy_parameters(retired.parameters));
        }
    }
    view.retired_features.clear();
    release(view.input_scratch);
    view.scratch_width = 0U;
    view.scratch_height = 0U;
    view.scratch_format = DXGI_FORMAT_UNKNOWN;
    release(view.sample_readback);
    view.samples_pending = false;
}

void evict_retired_features(ViewState& view) noexcept {
    while (view.retired_features.size() > retired_feature_capacity) {
        const auto retired = view.retired_features.front();
        view.retired_features.pop_front();
        if (retired.handle != nullptr && runtime.release_feature != nullptr) {
            static_cast<void>(runtime.release_feature(retired.handle));
        }
        if (retired.parameters != nullptr && runtime.destroy_parameters != nullptr) {
            static_cast<void>(runtime.destroy_parameters(retired.parameters));
        }
    }
}

[[nodiscard]] bool patch_slot(void** const slot, void* const replacement) noexcept {
    if (slot == nullptr || replacement == nullptr) return false;
    DWORD previous{};
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous)) {
        return false;
    }
    InterlockedExchangePointer(
        reinterpret_cast<PVOID volatile*>(slot),
        replacement
    );
    DWORD ignored{};
    static_cast<void>(VirtualProtect(
        slot,
        sizeof(*slot),
        previous,
        &ignored
    ));
    return true;
}

[[nodiscard]] bool patch_named_import(
    const HMODULE module,
    const char* const requested_name,
    void* const replacement,
    void** const original_output
) noexcept {
    if (module == nullptr || requested_name == nullptr || replacement == nullptr) {
        return false;
    }
    auto* const image = reinterpret_cast<std::byte*>(module);
    const auto* const dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return false;
    const auto* const headers = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
        image + dos->e_lfanew
    );
    if (headers->Signature != IMAGE_NT_SIGNATURE) return false;
    const auto& imports = headers->OptionalHeader.DataDirectory[
        IMAGE_DIRECTORY_ENTRY_IMPORT
    ];
    if (imports.VirtualAddress == 0U) return false;
    auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(
        image + imports.VirtualAddress
    );
    for (; descriptor->Name != 0U; ++descriptor) {
        if (descriptor->OriginalFirstThunk == 0U ||
            descriptor->FirstThunk == 0U) continue;
        auto* original = reinterpret_cast<IMAGE_THUNK_DATA64*>(
            image + descriptor->OriginalFirstThunk
        );
        auto* resolved = reinterpret_cast<IMAGE_THUNK_DATA64*>(
            image + descriptor->FirstThunk
        );
        for (; original->u1.AddressOfData != 0U; ++original, ++resolved) {
            if (IMAGE_SNAP_BY_ORDINAL64(original->u1.Ordinal)) continue;
            const auto* const import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(
                image + original->u1.AddressOfData
            );
            if (std::strcmp(
                    reinterpret_cast<const char*>(import->Name),
                    requested_name
                ) != 0) continue;
            auto* const slot = reinterpret_cast<void**>(&resolved->u1.Function);
            if (original_output != nullptr) *original_output = *slot;
            return patch_slot(slot, replacement);
        }
    }
    return false;
}

DWORD WINAPI hook_nr_get_module_file_name(
    const HMODULE module,
    LPWSTR const output,
    const DWORD capacity
) noexcept {
    // DLSS-NR 310.8 checks the identity of its external caller. Match the
    // working bridge implementation and present this add-on as nvngx.dll.
    if (module == runtime.addon && output != nullptr && capacity != 0U) {
        constexpr wchar_t identity[] = L"nvngx.dll";
        constexpr DWORD length = static_cast<DWORD>(std::size(identity) - 1U);
        if (capacity <= length) {
            output[0] = L'\0';
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return capacity;
        }
        std::memcpy(output, identity, sizeof(identity));
        return length;
    }
    return runtime.get_module_file_name == nullptr
        ? 0U
        : runtime.get_module_file_name(module, output, capacity);
}

[[nodiscard]] bool addon_directory(
    std::array<wchar_t, 32768U>& directory
) noexcept {
    HMODULE addon{};
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&evaluate_dlss_nr),
            &addon
        )) return false;
    runtime.addon = addon;
    const auto length = GetModuleFileNameW(
        addon,
        directory.data(),
        static_cast<DWORD>(directory.size())
    );
    if (length == 0U || length >= directory.size()) return false;
    for (DWORD index = length; index > 0U; --index) {
        if (directory[index - 1U] == L'\\' || directory[index - 1U] == L'/') {
            directory[index - 1U] = L'\0';
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool load_nr_module() noexcept {
    if (runtime.module != nullptr) return true;
    if (runtime.state == 2U || runtime.state == 3U) return false;
    if (runtime_load_started.exchange(true, std::memory_order_acq_rel)) {
        return runtime.module != nullptr;
    }

    std::array<wchar_t, 32768U> directory{};
    if (!addon_directory(directory)) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        trace_event("DLSS-NR runtime initialization failed: add-on directory unavailable");
        return false;
    }
    std::array<wchar_t, 32768U> path = directory;
    constexpr wchar_t filename[] = L"\\nvngx_dlssnr.dll";
    const auto used = std::wcslen(path.data());
    if (used + std::size(filename) > path.size()) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        return false;
    }
    std::memcpy(path.data() + used, filename, sizeof(filename));
    trace_event("DLSS-NR loading nvngx_dlssnr.dll off the NGX evaluate thread");
    runtime.module = LoadLibraryExW(
        path.data(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    );
    if (runtime.module == nullptr) {
        runtime.state = 2U;
        diagnostics.state = DlssNrState::runtime_missing;
        diagnostics.last_result = GetLastError();
        trace_event(
            "DLSS-NR runtime missing path=%ls error=%lu",
            path.data(),
            static_cast<unsigned long>(GetLastError())
        );
        return false;
    }
    return true;
}

[[nodiscard]] bool initialize_runtime(ID3D12Device* const device) noexcept {
    if (!host_evaluate_succeeded.load(std::memory_order_acquire)) {
        return false;
    }
    if (runtime.state == 1U) return runtime.device == device;
    if (runtime.state == 2U || runtime.state == 3U || device == nullptr) {
        return false;
    }
    if (!load_nr_module() || runtime.module == nullptr) return false;

    void* original_get_module_file_name{};
    if (!patch_named_import(
            runtime.module,
            "GetModuleFileNameW",
            reinterpret_cast<void*>(&hook_nr_get_module_file_name),
            &original_get_module_file_name
        )) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        trace_event("DLSS-NR runtime identity import patch failed");
        return false;
    }
    runtime.get_module_file_name = reinterpret_cast<GetModuleFileNameWFn>(
        original_get_module_file_name
    );

    const auto initialize = reinterpret_cast<NgxInitExtFn>(GetProcAddress(
        runtime.module,
        "NVSDK_NGX_D3D12_Init_Ext"
    ));
    auto allocate = reinterpret_cast<NgxAllocateParametersFn>(GetProcAddress(
        runtime.module,
        "NVSDK_NGX_D3D12_AllocateParameters"
    ));
    auto destroy = reinterpret_cast<NgxDestroyParametersFn>(GetProcAddress(
        runtime.module,
        "NVSDK_NGX_D3D12_DestroyParameters"
    ));
    runtime.create_feature = reinterpret_cast<NgxCreateFeatureFn>(GetProcAddress(
        runtime.module,
        "NVSDK_NGX_D3D12_CreateFeature"
    ));
    runtime.evaluate_feature = reinterpret_cast<NgxEvaluateFeatureFn>(
        GetProcAddress(runtime.module, "NVSDK_NGX_D3D12_EvaluateFeature")
    );
    runtime.release_feature = reinterpret_cast<NgxReleaseFeatureFn>(GetProcAddress(
        runtime.module,
        "NVSDK_NGX_D3D12_ReleaseFeature"
    ));
    if (allocate == nullptr || destroy == nullptr) {
        const auto core = GetModuleHandleW(L"_nvngx.dll");
        if (core != nullptr) {
            allocate = reinterpret_cast<NgxAllocateParametersFn>(GetProcAddress(
                core,
                "NVSDK_NGX_D3D12_AllocateParameters"
            ));
            destroy = reinterpret_cast<NgxDestroyParametersFn>(GetProcAddress(
                core,
                "NVSDK_NGX_D3D12_DestroyParameters"
            ));
        }
    }
    if (initialize == nullptr || allocate == nullptr || destroy == nullptr ||
        runtime.create_feature == nullptr || runtime.evaluate_feature == nullptr ||
        runtime.release_feature == nullptr) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        trace_event("DLSS-NR runtime has an incomplete NGX export set");
        return false;
    }

    std::array<wchar_t, 32768U> directory{};
    if (!addon_directory(directory)) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        return false;
    }
    constexpr unsigned long long application_id = 0x0876232cULL;
    constexpr std::uint32_t ngx_sdk_version = 0x15U;
    const auto result = initialize(
        application_id,
        directory.data(),
        device,
        ngx_sdk_version,
        nullptr
    );
    diagnostics.last_result = result;
    if (!ngx_succeeded(result)) {
        runtime.state = 3U;
        diagnostics.state = DlssNrState::runtime_failed;
        trace_event("DLSS-NR Init_Ext failed result=0x%08X", result);
        return false;
    }
    runtime.allocate_parameters = allocate;
    runtime.destroy_parameters = destroy;
    runtime.device = device;
    runtime.device->AddRef();
    runtime.state = 1U;
    trace_event("DLSS-NR 310.8 feature-18 runtime initialized");
    return true;
}

NgxResult neural_scaling_ratio_callback(NgxParameters* const parameters) noexcept {
    if (parameters == nullptr) return 0xBAD00005U;
    parameters->Set("DLSSNR.ScalingRatio", 1.0F);
    parameters->Set("DLSSNR.Scale", 1.0F);
    parameters->Set("DLSSNR.Upscaling", 0U);
    return 1U;
}

[[nodiscard]] ViewState& find_or_create_view(const DlssViewId view_id) {
    for (auto& view : views) {
        if (view.view_id == view_id) return view;
    }
    views.push_back(ViewState{});
    views.back().view_id = view_id;
    return views.back();
}

[[nodiscard]] bool create_feature(
    ViewState& view,
    const DlssNrFrame& frame,
    const Settings& settings,
    const std::uint32_t working_width,
    const std::uint32_t working_height
) noexcept {
    const auto create_flags = nr_feature_create_flags(frame.create_flags);
    const FeatureKey key{
        working_width,
        working_height,
        working_width,
        working_height,
        create_flags,
    };
    if (view.handle != nullptr && view.has_key && view.key == key) return true;
    if (view.feature_failed && view.has_key && view.key == key) return false;
    for (auto iterator = view.retired_features.begin();
         iterator != view.retired_features.end(); ++iterator) {
        if (!(iterator->key == key)) continue;
        const CachedFeature current{view.key, view.parameters, view.handle};
        view.parameters = iterator->parameters;
        view.handle = iterator->handle;
        view.retired_features.erase(iterator);
        if (current.parameters != nullptr || current.handle != nullptr) {
            view.retired_features.push_back(current);
        }
        evict_retired_features(view);
        view.key = key;
        view.has_key = true;
        view.feature_failed = false;
        view.was_enabled = false;
        view.settings_signature = 0U;
        return true;
    }
    if (view.handle != nullptr || view.parameters != nullptr) {
        // Retain a bounded set of replaced features. This both avoids releasing
        // work still in flight and makes common on/off foveation sizes reusable.
        view.retired_features.push_back({view.key, view.parameters, view.handle});
        view.parameters = nullptr;
        view.handle = nullptr;
        evict_retired_features(view);
    }
    view.feature_failed = false;
    view.key = key;
    view.has_key = true;

    NgxParameters* parameters{};
    auto result = runtime.allocate_parameters(&parameters);
    if (!ngx_succeeded(result) || parameters == nullptr) {
        diagnostics.state = DlssNrState::feature_failed;
        diagnostics.last_result = result;
        ++diagnostics.failed_calls;
        view.feature_failed = true;
        return false;
    }
    const auto preset = settings.nr_preset == 0U ? 1U : settings.nr_preset;
    parameters->Set("Width", working_width);
    parameters->Set("Height", working_height);
    parameters->Set("OutWidth", working_width);
    parameters->Set("OutHeight", working_height);
    parameters->Set("DLSSNR.Width", working_width);
    parameters->Set("DLSSNR.Height", working_height);
    parameters->Set("DLSSNR.InputWidth", working_width);
    parameters->Set("DLSSNR.InputHeight", working_height);
    parameters->Set("DLSSNR.OutputWidth", working_width);
    parameters->Set("DLSSNR.OutputHeight", working_height);
    parameters->Set("DLSSNR.Output.Width", working_width);
    parameters->Set("DLSSNR.Output.Height", working_height);
    parameters->Set("DLSSNR.ScalingRatio", 1.0F);
    parameters->Set("DLSSNR.Scale", 1.0F);
    parameters->Set("DLSSNR.Upscaling", 0U);
    parameters->Set("DLSSNR.Enabled", 1U);
    parameters->Set("DLSSNR.Hint.Render.Preset", preset);
    parameters->Set(
        "DLSSNRComputeScalingRatioCallback",
        reinterpret_cast<void*>(
            reinterpret_cast<std::uintptr_t>(&neural_scaling_ratio_callback)
        )
    );
    parameters->Set("PerfQualityValue", 0U);
    parameters->Set("DLSS.Feature.Create.Flags", create_flags);
    parameters->Set("CreationNodeMask", 1U);
    parameters->Set("VisibilityNodeMask", 1U);

    NgxHandle* handle{};
    constexpr std::uint32_t neural_feature_id = 18U;
    result = runtime.create_feature(
        frame.command_list,
        neural_feature_id,
        parameters,
        &handle
    );
    diagnostics.last_result = result;
    if (!ngx_succeeded(result) || handle == nullptr) {
        if (handle != nullptr) static_cast<void>(runtime.release_feature(handle));
        static_cast<void>(runtime.destroy_parameters(parameters));
        diagnostics.state = DlssNrState::feature_failed;
        ++diagnostics.failed_calls;
        view.feature_failed = true;
        trace_event(
            "DLSS-NR feature 18 creation failed view=%llu input=%ux%u "
            "output=%ux%u flags=0x%X result=0x%08X",
            static_cast<unsigned long long>(frame.view_id),
            working_width,
            working_height,
            working_width,
            working_height,
            create_flags,
            result
        );
        return false;
    }
    view.parameters = parameters;
    view.handle = handle;
    view.key = key;
    view.has_key = true;
    view.feature_failed = false;
    view.was_enabled = false;
    view.settings_signature = 0U;
    trace_event(
        "DLSS-NR feature 18 created view=%llu input=%ux%u output=%ux%u "
        "scale=%.2f flags=0x%X game_flags=0x%X preset=%u",
        static_cast<unsigned long long>(frame.view_id),
        working_width,
        working_height,
        working_width,
        working_height,
        1.0,
        create_flags,
        frame.create_flags,
        preset
    );
    return true;
}

[[nodiscard]] bool initialize_gpu_resources(
    ID3D12GraphicsCommandList* const command_list,
    ID3D12Resource* const game_output,
    const NrRegion& region,
    const std::uint32_t working_width,
    const std::uint32_t working_height,
    GpuResources& gpu
) noexcept {
    ID3D12Device* device{};
    ID3DBlob* encoded{};
    ID3DBlob* decoded{};
    ID3DBlob* shader_errors{};
    ID3DBlob* serialized{};
    ID3DBlob* signature_errors{};
    const auto cleanup = [&]() noexcept {
        release(signature_errors);
        release(serialized);
        release(shader_errors);
        release(decoded);
        release(encoded);
        release(device);
    };
    auto fail = [&](const char* const stage, const HRESULT result) noexcept {
        if (shader_errors != nullptr && shader_errors->GetBufferPointer() != nullptr) {
            trace_event(
                "DLSS-NR shader errors %s",
                static_cast<const char*>(shader_errors->GetBufferPointer())
            );
        }
        cleanup();
        release_gpu(gpu);
        diagnostics.state = DlssNrState::unsupported_resources;
        diagnostics.last_result = static_cast<NgxResult>(result);
        ++diagnostics.failed_calls;
        trace_event("DLSS-NR GPU setup failed stage=%s hr=0x%08X", stage, result);
        return false;
    };

    auto result = command_list->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(result) || device == nullptr) return fail("GetDevice", result);
    const auto game_desc = game_output->GetDesc();
    if (!is_dlss_nr_output_compatible(game_desc)) {
        return fail("unsupported output", E_INVALIDARG);
    }

    gpu.game_output = game_output;
    gpu.game_output->AddRef();
    gpu.width = region.width;
    gpu.height = region.height;
    gpu.working_width = working_width;
    gpu.working_height = working_height;
    if (region.base_x + region.width > game_desc.Width ||
        region.base_y + region.height > game_desc.Height) {
        return fail("output region out of bounds", E_INVALIDARG);
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1U;
    heap.VisibleNodeMask = 1U;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = gpu.width;
    texture.Height = gpu.height;
    texture.DepthOrArraySize = 1U;
    texture.MipLevels = 1U;
    texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    texture.SampleDesc.Count = 1U;
    texture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    result = device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &texture,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&gpu.original_output)
    );
    if (FAILED(result)) return fail("CreateCommittedResource(original)", result);
    texture.Width = working_width;
    texture.Height = working_height;
    for (auto** destination : {&gpu.color_proxy, &gpu.neural_output}) {
        result = device->CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &texture,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(destination)
        );
        if (FAILED(result)) return fail("CreateCommittedResource", result);
    }
    gpu.original_output->SetName(L"Cheeky DLSS-NR original HDR output");
    gpu.color_proxy->SetName(L"Cheeky DLSS-NR color proxy");
    gpu.neural_output->SetName(L"Cheeky DLSS-NR neural output");

    D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap{};
    descriptor_heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptor_heap.NumDescriptors = 8U;
    descriptor_heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    result = device->CreateDescriptorHeap(
        &descriptor_heap,
        IID_PPV_ARGS(&gpu.descriptors)
    );
    if (FAILED(result)) return fail("CreateDescriptorHeap", result);
    gpu.descriptor_size = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
    );
    auto cpu = gpu.descriptors->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC game_srv{};
    game_srv.Format = game_desc.Format;
    game_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    game_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    game_srv.Texture2D.MipLevels = 1U;
    device->CreateShaderResourceView(game_output, &game_srv, cpu);
    cpu.ptr += gpu.descriptor_size;
    D3D12_SHADER_RESOURCE_VIEW_DESC fp16_srv{};
    fp16_srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    fp16_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    fp16_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    fp16_srv.Texture2D.MipLevels = 1U;
    for (auto* resource : {
            gpu.original_output,
            gpu.color_proxy,
            gpu.neural_output,
        }) {
        device->CreateShaderResourceView(resource, &fp16_srv, cpu);
        cpu.ptr += gpu.descriptor_size;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC fp16_uav{};
    fp16_uav.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    fp16_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(
        gpu.original_output,
        nullptr,
        &fp16_uav,
        cpu
    );
    cpu.ptr += gpu.descriptor_size;
    device->CreateUnorderedAccessView(gpu.color_proxy, nullptr, &fp16_uav, cpu);
    cpu.ptr += gpu.descriptor_size;
    D3D12_UNORDERED_ACCESS_VIEW_DESC game_uav{};
    game_uav.Format = game_desc.Format;
    game_uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(game_output, nullptr, &game_uav, cpu);
    cpu.ptr += gpu.descriptor_size;
    device->CreateUnorderedAccessView(game_output, nullptr, &game_uav, cpu);

    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3U;
    ranges[0].BaseShaderRegister = 0U;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 2U;
    ranges[1].BaseShaderRegister = 0U;
    D3D12_ROOT_PARAMETER root_parameters[3]{};
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[0].DescriptorTable.NumDescriptorRanges = 1U;
    root_parameters[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[1].DescriptorTable.NumDescriptorRanges = 1U;
    root_parameters[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    root_parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_parameters[2].Constants.ShaderRegister = 0U;
    root_parameters[2].Constants.Num32BitValues = 21U;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3U;
    root_desc.pParameters = root_parameters;
    result = D3D12SerializeRootSignature(
        &root_desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized,
        &signature_errors
    );
    if (FAILED(result)) return fail("D3D12SerializeRootSignature", result);
    result = device->CreateRootSignature(
        0U,
        serialized->GetBufferPointer(),
        serialized->GetBufferSize(),
        IID_PPV_ARGS(&gpu.root_signature)
    );
    if (FAILED(result)) return fail("CreateRootSignature", result);

    constexpr char shader[] = R"(
Texture2D<float4> Source0 : register(t0);
Texture2D<float4> Source1 : register(t1);
Texture2D<float4> Source2 : register(t2);
RWTexture2D<float4> Output0 : register(u0);
RWTexture2D<float4> Output1 : register(u1);

cbuffer CodecConstants : register(b0) {
    uint2 Size;
    uint2 SourceSize;
    uint2 SourceBase;
    uint2 ProxySize;
    float PaperWhiteScale;
    float TransferStrength;
    float ColorStrength;
    uint HdrMode;
    uint2 RegionBase;
    uint2 RegionSize;
    float FoveationWidth;
    float FoveationHeight;
    float FoveationRoundness;
    float FoveationFeather;
    uint ShowAlignmentBorder;
};

float FoveationShapeDistance(float2 pixel) {
    const float2 centered =
        (pixel - float2(RegionBase) + 0.5) /
        (0.5 * max(float2(RegionSize), 1.0)) - 1.0;
    const float2 scaled = abs(centered);
    return lerp(
        max(scaled.x, scaled.y),
        length(scaled),
        saturate(FoveationRoundness)
    );
}

float Luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float4 LoadSource0Bilinear(float2 position, uint2 origin, uint2 dimensions) {
    const float2 base = floor(position);
    const float2 fraction = position - base;
    const int2 minimum = int2(origin);
    const int2 maximum = minimum + int2(dimensions) - 1;
    const int2 p00 = clamp(int2(base), minimum, maximum);
    const int2 p10 = clamp(p00 + int2(1, 0), minimum, maximum);
    const int2 p01 = clamp(p00 + int2(0, 1), minimum, maximum);
    const int2 p11 = clamp(p00 + int2(1, 1), minimum, maximum);
    return lerp(
        lerp(Source0.Load(int3(p00, 0)), Source0.Load(int3(p10, 0)), fraction.x),
        lerp(Source0.Load(int3(p01, 0)), Source0.Load(int3(p11, 0)), fraction.x),
        fraction.y
    );
}

float4 LoadSource1Bilinear(float2 position, uint2 dimensions) {
    const float2 base = floor(position);
    const float2 fraction = position - base;
    const int2 maximum = int2(dimensions) - 1;
    const int2 p00 = clamp(int2(base), int2(0, 0), maximum);
    const int2 p10 = clamp(p00 + int2(1, 0), int2(0, 0), maximum);
    const int2 p01 = clamp(p00 + int2(0, 1), int2(0, 0), maximum);
    const int2 p11 = clamp(p00 + int2(1, 1), int2(0, 0), maximum);
    return lerp(
        lerp(Source1.Load(int3(p00, 0)), Source1.Load(int3(p10, 0)), fraction.x),
        lerp(Source1.Load(int3(p01, 0)), Source1.Load(int3(p11, 0)), fraction.x),
        fraction.y
    );
}

float4 LoadSource2Bilinear(float2 position, uint2 dimensions) {
    const float2 base = floor(position);
    const float2 fraction = position - base;
    const int2 maximum = int2(dimensions) - 1;
    const int2 p00 = clamp(int2(base), int2(0, 0), maximum);
    const int2 p10 = clamp(p00 + int2(1, 0), int2(0, 0), maximum);
    const int2 p01 = clamp(p00 + int2(0, 1), int2(0, 0), maximum);
    const int2 p11 = clamp(p00 + int2(1, 1), int2(0, 0), maximum);
    return lerp(
        lerp(Source2.Load(int3(p00, 0)), Source2.Load(int3(p10, 0)), fraction.x),
        lerp(Source2.Load(int3(p01, 0)), Source2.Load(int3(p11, 0)), fraction.x),
        fraction.y
    );
}

float SampleLuma1(float2 position, uint2 dimensions) {
    return Luminance(max(LoadSource1Bilinear(position, dimensions).rgb, 0.0));
}

float SampleLuma2(float2 position, uint2 dimensions) {
    return Luminance(max(LoadSource2Bilinear(position, dimensions).rgb, 0.0));
}

float LocalLuma1(float2 position, uint2 dimensions) {
    float sum = SampleLuma1(position, dimensions);
    sum += SampleLuma1(position + float2(3.0, 0.0), dimensions);
    sum += SampleLuma1(position + float2(-3.0, 0.0), dimensions);
    sum += SampleLuma1(position + float2(0.0, 3.0), dimensions);
    sum += SampleLuma1(position + float2(0.0, -3.0), dimensions);
    return sum * 0.2;
}

float LocalLuma2(float2 position, uint2 dimensions) {
    float sum = SampleLuma2(position, dimensions);
    sum += SampleLuma2(position + float2(3.0, 0.0), dimensions);
    sum += SampleLuma2(position + float2(-3.0, 0.0), dimensions);
    sum += SampleLuma2(position + float2(0.0, 3.0), dimensions);
    sum += SampleLuma2(position + float2(0.0, -3.0), dimensions);
    return sum * 0.2;
}

[numthreads(16, 16, 1)]
void EncodeMain(uint3 dispatch_id : SV_DispatchThreadID) {
    if (all(dispatch_id.xy < Size)) {
        Output0[dispatch_id.xy] = Source0.Load(int3(SourceBase + dispatch_id.xy, 0));
    }
    if (all(dispatch_id.xy < ProxySize)) {
        const float2 source_position =
            float2(SourceBase) +
            (float2(dispatch_id.xy) + 0.5) * float2(SourceSize) /
            float2(ProxySize) - 0.5;
        const float4 proxy_source = LoadSource0Bilinear(
            source_position,
            SourceBase,
            SourceSize
        );
        const float3 linear_color = max(proxy_source.rgb, 0.0);
        const float luma = max(Luminance(linear_color), 1e-4);
        const float compressed = luma / (1.0 + luma);
        Output1[dispatch_id.xy] = float4(
            linear_color * (compressed / luma),
            1.0
        );
    }
}

[numthreads(16, 16, 1)]
void DecodeMain(uint3 dispatch_id : SV_DispatchThreadID) {
    if (any(dispatch_id.xy >= Size)) return;
    const float4 original_sample = Source0.Load(int3(dispatch_id.xy, 0));
    const bool inside_region = all(dispatch_id.xy >= RegionBase) &&
        all(dispatch_id.xy < RegionBase + RegionSize);
    const float distance_from_center = FoveationShapeDistance(dispatch_id.xy);
    const float distance_per_pixel = max(
        abs(FoveationShapeDistance(float2(dispatch_id.xy) + float2(1.0, 0.0)) -
            distance_from_center),
        abs(FoveationShapeDistance(float2(dispatch_id.xy) + float2(0.0, 1.0)) -
            distance_from_center)
    );
    const float normalized_feather = FoveationFeather /
        max(0.0001, min(FoveationWidth, FoveationHeight));
    const float foveation_weight = FoveationFeather <= 0.0
        ? (distance_from_center <= 1.0 ? 1.0 : 0.0)
        : 1.0 - smoothstep(
            max(0.0, 1.0 - normalized_feather),
            1.0,
            distance_from_center
        );
    if (!inside_region || foveation_weight <= 0.0) {
        Output0[SourceBase + dispatch_id.xy] = original_sample;
        return;
    }
    const float2 proxy_position =
        (float2(dispatch_id.xy - RegionBase) + 0.5) * float2(ProxySize) /
        float2(RegionSize) - 0.5;
    const float4 proxy_sample = LoadSource1Bilinear(proxy_position, ProxySize);
    const float4 neural_sample = LoadSource2Bilinear(proxy_position, ProxySize);
    const bool neural_bad = any(neural_sample.rgb != neural_sample.rgb) ||
        any(abs(neural_sample.rgb) > 65504.0);
    const float original_y = Luminance(max(original_sample.rgb, 0.0));
    const float proxy_y = Luminance(max(proxy_sample.rgb, 0.0));
    const float neural_y = Luminance(max(neural_sample.rgb, 0.0));
    const float local_proxy = LocalLuma1(proxy_position, ProxySize);
    const float local_neural = LocalLuma2(proxy_position, ProxySize);
    const bool neural_in_range = !neural_bad &&
        original_y > 1e-5 && proxy_y > 1e-5 && neural_y > 1e-5 &&
        abs(local_neural - local_proxy) < 0.12;
    if (ShowAlignmentBorder != 0U) {
        const float ring_y = max(original_y, 0.02);
        if (distance_from_center <= 1.0 &&
            distance_from_center >= 1.0 - 5.0 * distance_per_pixel) {
            Output0[SourceBase + dispatch_id.xy] = float4(
                original_sample.rgb * float3(0.05, 1.0, 0.05) +
                    float3(0.0, ring_y * 3.0, 0.0),
                1.0
            );
            return;
        }
        if (distance_from_center < 1.0 - 5.0 * distance_per_pixel &&
            distance_from_center >= 1.0 - 10.0 * distance_per_pixel) {
            const float3 tint = neural_in_range
                ? float3(1.0, 1.0, 0.05)
                : float3(1.0, 0.05, 1.0);
            Output0[SourceBase + dispatch_id.xy] = float4(
                original_sample.rgb * tint * 0.2 + tint * ring_y * 3.0,
                1.0
            );
            return;
        }
    }
    if (!neural_in_range) {
        Output0[SourceBase + dispatch_id.xy] = original_sample;
        return;
    }
    const float proxy_hf = proxy_y - local_proxy;
    const float neural_hf = neural_y - local_neural;
    const float hf = clamp(neural_hf - proxy_hf, -0.06, 0.06);
    const float orig_r = original_y / (1.0 + original_y);
    const float new_r = saturate(
        orig_r + hf * saturate(ColorStrength) * foveation_weight
    );
    const float new_y = new_r / max(1.0 - new_r, 1e-4);
    const float3 processed =
        original_sample.rgb * (new_y / max(original_y, 1e-4));
    Output0[SourceBase + dispatch_id.xy] = float4(processed, original_sample.a);
}
)";

    result = D3DCompile(
        shader,
        sizeof(shader) - 1U,
        "Cheeky DLSS-NR codec",
        nullptr,
        nullptr,
        "EncodeMain",
        "cs_5_1",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0U,
        &encoded,
        &shader_errors
    );
    if (FAILED(result)) return fail("D3DCompile(encode)", result);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = gpu.root_signature;
    pipeline.CS = {encoded->GetBufferPointer(), encoded->GetBufferSize()};
    result = device->CreateComputePipelineState(
        &pipeline,
        IID_PPV_ARGS(&gpu.encode_pipeline)
    );
    if (FAILED(result)) return fail("CreateComputePipelineState(encode)", result);
    release(shader_errors);
    result = D3DCompile(
        shader,
        sizeof(shader) - 1U,
        "Cheeky DLSS-NR codec",
        nullptr,
        nullptr,
        "DecodeMain",
        "cs_5_1",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0U,
        &decoded,
        &shader_errors
    );
    if (FAILED(result)) return fail("D3DCompile(decode)", result);
    pipeline.CS = {decoded->GetBufferPointer(), decoded->GetBufferSize()};
    result = device->CreateComputePipelineState(
        &pipeline,
        IID_PPV_ARGS(&gpu.decode_pipeline)
    );
    if (FAILED(result)) return fail("CreateComputePipelineState(decode)", result);
    cleanup();
    trace_event(
        "DLSS-NR region codec created region=%ux%u working=%ux%u source=%u,%u "
        "format=0x%X",
        gpu.width,
        gpu.height,
        gpu.working_width,
        gpu.working_height,
        region.base_x,
        region.base_y,
        static_cast<unsigned>(game_desc.Format)
    );
    return true;
}

[[nodiscard]] GpuResources* find_or_create_gpu(
    ViewState& view,
    const DlssNrFrame& frame,
    const NrRegion& region,
    const std::uint32_t working_width,
    const std::uint32_t working_height
) noexcept {
    for (auto& gpu : view.gpu_resources) {
        if (gpu.game_output == frame.color && gpu.width == region.width &&
            gpu.height == region.height && gpu.working_width == working_width &&
            gpu.working_height == working_height) return &gpu;
    }
    view.gpu_resources.push_back(GpuResources{});
    auto& gpu = view.gpu_resources.back();
    if (!initialize_gpu_resources(
            frame.command_list,
            frame.color,
            region,
            working_width,
            working_height,
            gpu
        )) {
        view.gpu_resources.pop_back();
        return nullptr;
    }
    while (view.gpu_resources.size() > gpu_resource_cache_capacity) {
        release_gpu(view.gpu_resources.front());
        view.gpu_resources.pop_front();
    }
    return &gpu;
}

[[nodiscard]] NrRegion calculate_region(
    const Settings& settings,
    const std::uint32_t width,
    const std::uint32_t height,
    const FoveationGeometry* const shared_sr_crop,
    const std::uint32_t render_width,
    const std::uint32_t render_height,
    const std::uint32_t travel_width = 0U,
    const std::uint32_t travel_height = 0U,
    const std::uint32_t eye_base_x = 0U,
    const std::uint32_t eye_base_y = 0U
) noexcept {
    if (!settings.nr_foveated) {
        Settings full_view = settings;
        full_view.nr_width = 1.0F;
        full_view.nr_height = 1.0F;
        full_view.nr_center_x = 0.0F;
        full_view.nr_x_offset = 0.0F;
        full_view.nr_height_offset = 0.0F;
        full_view.nr_source_x = 0.0F;
        full_view.nr_source_y = 0.0F;
        full_view.nr_use_sr_foveation = false;
        const auto crop = calculate_dlss_nr_view_crop(
            full_view,
            width,
            height,
            nullptr,
            0U,
            0U,
            travel_width,
            travel_height,
            eye_base_x,
            eye_base_y
        );
        return {
            crop.origin_x,
            crop.origin_y,
            crop.width,
            crop.height,
            1.0F,
            1.0F,
            0.0F,
            0.0F,
        };
    }
    const auto crop = calculate_dlss_nr_view_crop(
        settings,
        width,
        height,
        shared_sr_crop,
        render_width,
        render_height,
        travel_width,
        travel_height,
        eye_base_x,
        eye_base_y
    );
    const auto parameters = dlss_nr_foveation_parameters(
        settings, shared_sr_crop, render_width, render_height
    );
    return {
        crop.origin_x,
        crop.origin_y,
        crop.width,
        crop.height,
        parameters.width,
        parameters.height,
        parameters.roundness,
        parameters.transition_width,
    };
}

[[nodiscard]] std::uint32_t scaled_extent(
    const std::uint32_t extent,
    const float scale
) noexcept {
    const auto requested = (std::max)(
        32U,
        static_cast<std::uint32_t>(
            static_cast<float>(extent) * std::clamp(scale, 0.1F, 1.0F) + 0.5F
        )
    );
    return (requested + 7U) / 8U * 8U;
}

struct ScaledSubrect {
    std::uint32_t base{};
    std::uint32_t extent{};
};

[[nodiscard]] ScaledSubrect scale_subrect(
    const std::uint32_t region_base,
    const std::uint32_t region_extent,
    const std::uint32_t source_base,
    const std::uint32_t source_extent,
    const std::uint32_t output_extent
) noexcept {
    const auto base = static_cast<std::uint32_t>(std::floor(
        static_cast<double>(region_base) * source_extent / output_extent
    ));
    const auto end = (std::min)(
        source_extent,
        static_cast<std::uint32_t>(std::ceil(
            static_cast<double>(region_base + region_extent) * source_extent /
                output_extent
        ))
    );
    return {source_base + base, (std::max)(1U, end - base)};
}

[[nodiscard]] std::uint64_t settings_signature(
    const Settings& settings,
    const NrRegion& region
) noexcept {
    std::uint64_t signature = 1469598103934665603ULL;
    const auto append = [&signature](const std::uint32_t value) noexcept {
        signature ^= value;
        signature *= 1099511628211ULL;
    };
    for (const auto value : {
            settings.nr_working_scale,
            settings.nr_intensity,
            settings.nr_local_tone_strength,
            settings.nr_local_structure_strength,
            settings.nr_skin_structure_strength,
            settings.nr_paper_white_scale,
            settings.nr_hdr_transfer_strength,
            settings.nr_color_strength,
            settings.nr_motion_scale_x_multiplier,
            settings.nr_motion_scale_y_multiplier,
            region.shape_width,
            region.shape_height,
            region.roundness,
            region.transition,
        }) {
        std::uint32_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        append(bits);
    }
    append(settings.nr_foveated ? 1U : 0U);
    append(settings.nr_automatic_mask ? 1U : 0U);
    append(settings.nr_ui_correction ? 1U : 0U);
    append(settings.nr_depth_convention);
    append(region.base_x);
    append(region.base_y);
    append(region.width);
    append(region.height);
    return signature;
}

void transition(
    ID3D12GraphicsCommandList* const command_list,
    ID3D12Resource* const resource,
    const D3D12_RESOURCE_STATES before,
    const D3D12_RESOURCE_STATES after
) noexcept {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    command_list->ResourceBarrier(1U, &barrier);
}

void restore_original_color(
    const DlssNrFrame& frame,
    GpuResources& gpu,
    const NrRegion& codec_region
) noexcept {
    const auto original_format = gpu.original_output->GetDesc().Format;
    const auto presented_format = frame.color->GetDesc().Format;
    if (original_format != presented_format) {
        transition(
            frame.command_list,
            gpu.original_output,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        transition(
            frame.command_list,
            gpu.color_proxy,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        transition(
            frame.command_list,
            frame.color,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            frame.color_state
        );
        return;
    }
    transition(
        frame.command_list,
        gpu.original_output,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    transition(
        frame.command_list,
        frame.color,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST
    );
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = gpu.original_output;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = frame.color;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    frame.command_list->CopyTextureRegion(
        &destination,
        codec_region.base_x,
        codec_region.base_y,
        0U,
        &source,
        nullptr
    );
    transition(
        frame.command_list,
        gpu.original_output,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    transition(
        frame.command_list,
        gpu.color_proxy,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    transition(
        frame.command_list,
        frame.color,
        D3D12_RESOURCE_STATE_COPY_DEST,
        frame.color_state
    );
}

void uav_barrier(
    ID3D12GraphicsCommandList* const command_list,
    ID3D12Resource* const resource
) noexcept {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    command_list->ResourceBarrier(1U, &barrier);
}

void copy_texture_region(
    ID3D12GraphicsCommandList* const command_list,
    ID3D12Resource* const destination,
    const std::uint32_t dest_x,
    const std::uint32_t dest_y,
    ID3D12Resource* const source,
    const std::uint32_t src_x,
    const std::uint32_t src_y,
    const std::uint32_t width,
    const std::uint32_t height
) noexcept {
    D3D12_TEXTURE_COPY_LOCATION dest_location{};
    dest_location.pResource = destination;
    dest_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dest_location.SubresourceIndex = 0U;
    D3D12_TEXTURE_COPY_LOCATION source_location{};
    source_location.pResource = source;
    source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source_location.SubresourceIndex = 0U;
    D3D12_BOX box{};
    box.left = src_x;
    box.top = src_y;
    box.front = 0U;
    box.right = src_x + width;
    box.bottom = src_y + height;
    box.back = 1U;
    command_list->CopyTextureRegion(
        &dest_location, dest_x, dest_y, 0U, &source_location, &box
    );
}

[[nodiscard]] bool ensure_input_scratch(
    ViewState& view,
    ID3D12Device* const device,
    const std::uint32_t width,
    const std::uint32_t height,
    const DXGI_FORMAT format
) noexcept {
    if (view.input_scratch != nullptr &&
        view.scratch_width == width &&
        view.scratch_height == height &&
        view.scratch_format == format) {
        return true;
    }
    release(view.input_scratch);
    view.scratch_width = 0U;
    view.scratch_height = 0U;
    view.scratch_format = DXGI_FORMAT_UNKNOWN;
    if (device == nullptr || width == 0U || height == 0U) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1U;
    heap.VisibleNodeMask = 1U;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = 1U;
    texture.MipLevels = 1U;
    texture.Format = format;
    texture.SampleDesc.Count = 1U;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto result = device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &texture,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&view.input_scratch)
    );
    if (FAILED(result)) {
        trace_event(
            "NR before SR scratch create failed hr=0x%08X format=%u %ux%u",
            static_cast<unsigned>(result),
            static_cast<unsigned>(format),
            width,
            height
        );
        return false;
    }
    view.scratch_width = width;
    view.scratch_height = height;
    view.scratch_format = format;
    return true;
}

void dispatch_codec(
    const DlssNrFrame& frame,
    GpuResources& gpu,
    ID3D12PipelineState* const pipeline,
    const std::uint32_t source_descriptor,
    const std::uint32_t destination_descriptor,
    const Settings& settings,
    const NrRegion& region,
    const std::uint32_t hdr_mode
) noexcept {
    ID3D12DescriptorHeap* heaps[]{gpu.descriptors};
    frame.command_list->SetDescriptorHeaps(1U, heaps);
    frame.command_list->SetComputeRootSignature(gpu.root_signature);
    frame.command_list->SetPipelineState(pipeline);
    auto handle = gpu.descriptors->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<std::uint64_t>(source_descriptor) *
        gpu.descriptor_size;
    frame.command_list->SetComputeRootDescriptorTable(0U, handle);
    handle = gpu.descriptors->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<std::uint64_t>(destination_descriptor) *
        gpu.descriptor_size;
    frame.command_list->SetComputeRootDescriptorTable(1U, handle);
    struct CodecConstants {
        std::uint32_t size[2];
        std::uint32_t source_size[2];
        std::uint32_t source_base[2];
        std::uint32_t proxy_size[2];
        float paper_white_scale;
        float transfer_strength;
        float color_strength;
        std::uint32_t hdr_mode;
        std::uint32_t region_base[2];
        std::uint32_t region_size[2];
        float foveation_width;
        float foveation_height;
        float foveation_roundness;
        float foveation_feather;
        std::uint32_t show_alignment_border;
    };
    static_assert(sizeof(CodecConstants) == 21U * sizeof(std::uint32_t));
    const CodecConstants constants{
        {gpu.width, gpu.height},
        {gpu.width, gpu.height},
        {region.base_x, region.base_y},
        {gpu.working_width, gpu.working_height},
        settings.nr_paper_white_scale,
        settings.nr_hdr_transfer_strength,
        settings.nr_color_strength,
        hdr_mode,
        {0U, 0U},
        {gpu.width, gpu.height},
        region.shape_width,
        region.shape_height,
        region.roundness,
        region.transition,
        settings.nr_alignment_border_enabled ? 1U : 0U,
    };
    frame.command_list->SetComputeRoot32BitConstants(2U, 21U, &constants, 0U);
    const auto dispatch_width = (std::max)(gpu.width, gpu.working_width);
    const auto dispatch_height = (std::max)(gpu.height, gpu.working_height);
    frame.command_list->Dispatch(
        (dispatch_width + 15U) / 16U,
        (dispatch_height + 15U) / 16U,
        1U
    );
}

constexpr std::uint32_t color_sample_points = 5U;
constexpr std::uint32_t color_sample_buffers = 3U;
constexpr std::uint32_t color_sample_pitch = 256U;

[[nodiscard]] bool should_record_color_samples() noexcept {
    static std::atomic<std::uint32_t> sequence{};
    const auto sample = sequence.fetch_add(1U, std::memory_order_relaxed);
    return sample < 8U || sample % 150U == 0U;
}

[[nodiscard]] bool ensure_sample_readback(
    ViewState& view,
    ID3D12Device* const device
) noexcept {
    if (view.sample_readback != nullptr) return true;
    if (device == nullptr) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    heap.CreationNodeMask = 1U;
    heap.VisibleNodeMask = 1U;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = static_cast<std::uint64_t>(color_sample_points) *
        color_sample_buffers * color_sample_pitch;
    desc.Height = 1U;
    desc.DepthOrArraySize = 1U;
    desc.MipLevels = 1U;
    desc.SampleDesc.Count = 1U;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const auto result = device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&view.sample_readback)
    );
    return SUCCEEDED(result) && view.sample_readback != nullptr;
}

void copy_color_sample_pixel(
    ID3D12GraphicsCommandList* const command_list,
    ID3D12Resource* const readback,
    const std::uint32_t slot,
    ID3D12Resource* const source,
    const std::uint32_t x,
    const std::uint32_t y
) noexcept {
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Offset =
        static_cast<std::uint64_t>(slot) * color_sample_pitch;
    destination.PlacedFootprint.Footprint.Format =
        DXGI_FORMAT_R16G16B16A16_FLOAT;
    destination.PlacedFootprint.Footprint.Width = 1U;
    destination.PlacedFootprint.Footprint.Height = 1U;
    destination.PlacedFootprint.Footprint.Depth = 1U;
    destination.PlacedFootprint.Footprint.RowPitch = color_sample_pitch;
    D3D12_TEXTURE_COPY_LOCATION source_location{};
    source_location.pResource = source;
    source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source_location.SubresourceIndex = 0U;
    D3D12_BOX box{};
    box.left = x;
    box.top = y;
    box.front = 0U;
    box.right = x + 1U;
    box.bottom = y + 1U;
    box.back = 1U;
    command_list->CopyTextureRegion(
        &destination, 0U, 0U, 0U, &source_location, &box
    );
}

void record_color_samples(
    ViewState& view,
    GpuResources& gpu,
    ID3D12GraphicsCommandList* const command_list,
    const Settings& settings,
    const DXGI_FORMAT color_format
) noexcept {
    if (!should_record_color_samples() || command_list == nullptr) return;
    if (!ensure_sample_readback(view, runtime.device)) return;
    const std::uint32_t crop_w = (std::max)(1U, gpu.width);
    const std::uint32_t crop_h = (std::max)(1U, gpu.height);
    const std::uint32_t proxy_w = (std::max)(1U, gpu.working_width);
    const std::uint32_t proxy_h = (std::max)(1U, gpu.working_height);
    const float u[color_sample_points]{0.15F, 0.85F, 0.50F, 0.15F, 0.85F};
    const float v[color_sample_points]{0.15F, 0.15F, 0.50F, 0.85F, 0.85F};
    auto sample_x = [](const float coord, const std::uint32_t extent) noexcept {
        const auto max_index = extent > 0U ? extent - 1U : 0U;
        return (std::min)(
            max_index,
            static_cast<std::uint32_t>(coord * static_cast<float>(max_index) + 0.5F)
        );
    };
    transition(
        command_list, gpu.original_output,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    transition(
        command_list, gpu.color_proxy,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    transition(
        command_list, gpu.neural_output,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    ID3D12Resource* const sources[color_sample_buffers]{
        gpu.original_output, gpu.color_proxy, gpu.neural_output
    };
    const std::uint32_t widths[color_sample_buffers]{crop_w, proxy_w, proxy_w};
    const std::uint32_t heights[color_sample_buffers]{crop_h, proxy_h, proxy_h};
    for (std::uint32_t point = 0U; point < color_sample_points; ++point) {
        for (std::uint32_t buffer = 0U; buffer < color_sample_buffers; ++buffer) {
            copy_color_sample_pixel(
                command_list,
                view.sample_readback,
                point * color_sample_buffers + buffer,
                sources[buffer],
                sample_x(u[point], widths[buffer]),
                sample_x(v[point], heights[buffer])
            );
        }
    }
    transition(
        command_list, gpu.original_output,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    transition(
        command_list, gpu.color_proxy,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    transition(
        command_list, gpu.neural_output,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    view.sample_crop_width = crop_w;
    view.sample_crop_height = crop_h;
    view.sample_proxy_width = proxy_w;
    view.sample_proxy_height = proxy_h;
    view.sample_format = static_cast<std::uint32_t>(color_format);
    view.sample_paper_white = settings.nr_paper_white_scale;
    view.samples_pending = true;
}

[[nodiscard]] float fp16_to_float(const std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1FU;
    const std::uint32_t mantissa = value & 0x3FFU;
    std::uint32_t bits = sign;
    if (exponent == 0U) {
        if (mantissa != 0U) {
            std::uint32_t mag = mantissa;
            std::uint32_t exp = 127U - 15U + 1U;
            while ((mag & 0x400U) == 0U) {
                mag <<= 1U;
                --exp;
            }
            bits |= (exp << 23U) | ((mag & 0x3FFU) << 13U);
        }
    } else if (exponent == 31U) {
        bits |= 0x7F800000U | (mantissa << 13U);
    } else {
        bits |= ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    float result{};
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void log_pending_color_samples(ViewState& view) noexcept {
    if (!view.samples_pending || view.sample_readback == nullptr) return;
    const auto size = static_cast<std::uint64_t>(color_sample_points) *
        color_sample_buffers * color_sample_pitch;
    D3D12_RANGE range{0U, size};
    void* mapped{};
    if (FAILED(view.sample_readback->Map(0U, &range, &mapped)) ||
        mapped == nullptr) {
        view.samples_pending = false;
        return;
    }
    struct BufferStats {
        float luma_min{1.0e30F};
        float luma_max{-1.0e30F};
        float luma_sum{};
        float max_abs{};
        float center_r{};
        float center_g{};
        float center_b{};
        std::uint32_t nan_count{};
        std::uint32_t inf_count{};
        std::uint32_t finite_count{};
    };
    BufferStats stats[color_sample_buffers]{};
    const auto* const bytes = static_cast<const std::uint8_t*>(mapped);
    for (std::uint32_t point = 0U; point < color_sample_points; ++point) {
        for (std::uint32_t buffer = 0U; buffer < color_sample_buffers; ++buffer) {
            const auto slot = point * color_sample_buffers + buffer;
            const auto* const pixel = reinterpret_cast<const std::uint16_t*>(
                bytes + static_cast<std::size_t>(slot) * color_sample_pitch
            );
            const float rgb[3]{
                fp16_to_float(pixel[0]),
                fp16_to_float(pixel[1]),
                fp16_to_float(pixel[2])
            };
            auto& stat = stats[buffer];
            const bool bad =
                rgb[0] != rgb[0] || rgb[1] != rgb[1] || rgb[2] != rgb[2];
            const bool inf =
                std::isinf(rgb[0]) || std::isinf(rgb[1]) || std::isinf(rgb[2]);
            if (bad) {
                ++stat.nan_count;
                continue;
            }
            if (inf) {
                ++stat.inf_count;
                continue;
            }
            ++stat.finite_count;
            const float luma =
                rgb[0] * 0.2126F + rgb[1] * 0.7152F + rgb[2] * 0.0722F;
            stat.luma_min = (std::min)(stat.luma_min, luma);
            stat.luma_max = (std::max)(stat.luma_max, luma);
            stat.luma_sum += luma;
            stat.max_abs = (std::max)(
                stat.max_abs,
                (std::max)(std::fabs(rgb[0]), (std::max)(std::fabs(rgb[1]), std::fabs(rgb[2])))
            );
            if (point == 2U) {
                stat.center_r = rgb[0];
                stat.center_g = rgb[1];
                stat.center_b = rgb[2];
            }
        }
    }
    D3D12_RANGE empty{};
    view.sample_readback->Unmap(0U, &empty);
    view.samples_pending = false;
    const auto describe = [](const BufferStats& stat) noexcept {
        if (stat.finite_count == 0U) return "empty";
        if (stat.max_abs < 1.0e-6F) return "cleared";
        if (stat.luma_max <= 1.2F) return "gamma-or-display-1";
        if (stat.luma_max <= 16.0F) return "linear-paper-white";
        return "scene-referred-or-scaled";
    };
    const char* names[color_sample_buffers]{"original", "proxy", "neural"};
    trace_event(
        "DLSS-NR color samples hdr_mode=0 paper_white=%.3f format=0x%X "
        "crop=%ux%u proxy=%ux%u",
        view.sample_paper_white,
        view.sample_format,
        view.sample_crop_width,
        view.sample_crop_height,
        view.sample_proxy_width,
        view.sample_proxy_height
    );
    for (std::uint32_t buffer = 0U; buffer < color_sample_buffers; ++buffer) {
        const auto& stat = stats[buffer];
        const auto mean = stat.finite_count == 0U
            ? 0.0F
            : stat.luma_sum / static_cast<float>(stat.finite_count);
        const auto luma_min = stat.finite_count == 0U ? 0.0F : stat.luma_min;
        const auto luma_max = stat.finite_count == 0U ? 0.0F : stat.luma_max;
        trace_event(
            "DLSS-NR %s luma min=%.4g max=%.4g mean=%.4g maxAbs=%.4g "
            "center=%.4g,%.4g,%.4g nan=%u inf=%u hint=%s",
            names[buffer],
            luma_min,
            luma_max,
            mean,
            stat.max_abs,
            stat.center_r,
            stat.center_g,
            stat.center_b,
            stat.nan_count,
            stat.inf_count,
            describe(stat)
        );
    }
}

}  // namespace

bool calculate_dlss_nr_geometry(
    const Settings& settings,
    const std::uint32_t output_width,
    const std::uint32_t output_height,
    DlssNrGeometry& geometry,
    const std::uint32_t travel_width,
    const std::uint32_t travel_height
) noexcept {
    if (output_width == 0U || output_height == 0U) return false;
    const auto region = calculate_region(
        settings, output_width, output_height, nullptr, 0U, 0U,
        travel_width, travel_height
    );
    if (region.width == 0U || region.height == 0U) return false;
    geometry = {
        region.base_x,
        region.base_y,
        region.width,
        region.height,
        scaled_extent(region.width, settings.nr_working_scale),
        scaled_extent(region.height, settings.nr_working_scale),
    };
    return true;
}

void note_dlss_nr_host_evaluate_succeeded() noexcept {
    host_evaluate_succeeded.store(true, std::memory_order_release);
}

void note_dlss_nr_host_device(ID3D12Device* const device) noexcept {
    if (device == nullptr) return;
    std::lock_guard lock(nr_mutex);
    if (pending_device == device) return;
    if (pending_device != nullptr) pending_device->Release();
    pending_device = device;
    pending_device->AddRef();
}

void pump_dlss_nr_runtime() noexcept {
    if (!host_evaluate_succeeded.load(std::memory_order_acquire)) return;
    if (!load_nr_module()) return;
    std::lock_guard lock(nr_mutex);
    if (runtime.state == 1U || runtime.state == 2U || runtime.state == 3U) {
        return;
    }
    auto* const device = pending_device;
    if (device == nullptr) return;
    static_cast<void>(initialize_runtime(device));
}

bool evaluate_dlss_nr(
    const DlssNrFrame& frame,
    const Settings& settings
) noexcept {
    Settings nr_settings = settings;
    if (frame.after_polish) {
        apply_nr_after_polish(nr_settings);
    }
    std::uint32_t color_width = 0U;
    std::uint32_t color_height = 0U;
    if (frame.color != nullptr) {
        const auto color_desc = frame.color->GetDesc();
        color_width = static_cast<std::uint32_t>(color_desc.Width);
        color_height = color_desc.Height;
    }
    std::uint32_t view_width = frame.output_width;
    std::uint32_t view_height = frame.output_height;
    std::uint32_t travel_width = view_width;
    std::uint32_t travel_height = view_height;
    if (frame.pre_upscale) {
        const auto canvas = dlss_nr_pre_upscale_canvas(
            frame.input_width,
            frame.input_height,
            frame.output_width,
            frame.output_height,
            color_width,
            color_height
        );
        view_width = frame.input_width;
        view_height = frame.input_height;
        travel_width = canvas.travel_width;
        travel_height = canvas.travel_height;
    } else if (color_width != 0U) {
        travel_width = (std::max)(view_width, color_width);
        travel_height = (std::max)(view_height, color_height);
    }
    const auto eye_base_x = frame.color_is_region ? 0U : frame.color_base_x;
    const auto eye_base_y = frame.color_is_region ? 0U : frame.color_base_y;
    if (frame.color != nullptr) {
        apply_openxr_gaze_to_nr_settings(
            nr_settings,
            frame.view_id,
            frame.color,
            eye_base_x,
            eye_base_y,
            view_width,
            view_height,
            travel_width,
            travel_height,
            true
        );
    }
    if (frame.pre_upscale && frame.output_width != 0U &&
        frame.output_height != 0U) {
        nr_settings.nr_source_x *= static_cast<float>(frame.input_width) /
            static_cast<float>(frame.output_width);
        nr_settings.nr_source_y *= static_cast<float>(frame.input_height) /
            static_cast<float>(frame.output_height);
    }
    std::unique_lock lock(nr_mutex);
    diagnostics.route = frame.route;
    if (!settings.nr_enabled) {
        diagnostics.state = DlssNrState::disabled;
        for (auto& view : views) view.was_enabled = false;
        return false;
    }
    ++diagnostics.candidate_calls;
    if (frame.command_list == nullptr || frame.view_id == 0U ||
        frame.color == nullptr || frame.depth == nullptr ||
        frame.motion_vectors == nullptr || frame.input_width == 0U ||
        frame.input_height == 0U || frame.output_width == 0U ||
        frame.output_height == 0U || frame.depth_width == 0U ||
        frame.depth_height == 0U || frame.motion_width == 0U ||
        frame.motion_height == 0U) {
        diagnostics.state = DlssNrState::unsupported_resources;
        ++diagnostics.failed_calls;
        trace_event(
            "DLSS-NR missing resources view=%llu color=%p depth=%p mv=%p "
            "input=%ux%u output=%ux%u depth=%ux%u mv=%ux%u",
            static_cast<unsigned long long>(frame.view_id),
            frame.color,
            frame.depth,
            frame.motion_vectors,
            frame.input_width,
            frame.input_height,
            frame.output_width,
            frame.output_height,
            frame.depth_width,
            frame.depth_height,
            frame.motion_width,
            frame.motion_height
        );
        return false;
    }
    ID3D12Device* device{};
    const auto device_result = frame.command_list->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(device_result) || device == nullptr) {
        diagnostics.state = DlssNrState::unsupported_resources;
        diagnostics.last_result = static_cast<NgxResult>(device_result);
        ++diagnostics.failed_calls;
        return false;
    }
    const bool runtime_ready =
        runtime.state == 1U && same_d3d12_device(runtime.device, device);
    if (!runtime_ready) {
        if (pending_device != device) {
            if (pending_device != nullptr) pending_device->Release();
            pending_device = device;
            pending_device->AddRef();
        }
        device->Release();
        const auto waits = diagnostics.failed_calls;
        if (waits < 8U || waits % 300U == 0U) {
            trace_event(
                "DLSS-NR waiting for runtime host_evaluate=%s pending=%p "
                "state=%u",
                host_evaluate_succeeded.load(std::memory_order_acquire)
                    ? "yes"
                    : "no",
                pending_device,
                runtime.state
            );
        }
        ++diagnostics.failed_calls;
        return false;
    }

    DlssNrFrame work = frame;
    if (work.pre_upscale) {
        work.view_id ^= 0xA11AF0E500000001ULL;
    } else if (work.after_polish) {
        work.view_id ^= 0xA11AF0E500000002ULL;
    }
    const auto color_desc = work.color->GetDesc();
    auto region = calculate_region(
        nr_settings,
        view_width,
        view_height,
        work.has_shared_sr_crop ? &work.shared_sr_crop : nullptr,
        work.input_width,
        work.input_height,
        travel_width,
        travel_height,
        eye_base_x,
        eye_base_y
    );
    auto& view = find_or_create_view(work.view_id);
    const auto stabilize = [](
        const std::uint32_t requested,
        const std::uint32_t previous,
        const bool has_previous
    ) noexcept {
        if (!has_previous || previous < 8U) return requested;
        const auto delta = requested > previous
            ? requested - previous : previous - requested;
        return delta <= 1U ? previous : requested;
    };
    region.width = stabilize(region.width, view.key.input_width, view.has_key);
    region.height = stabilize(
        region.height, view.key.input_height, view.has_key
    );
    const auto canvas_width = work.pre_upscale
        ? (std::min)(
            static_cast<std::uint32_t>(color_desc.Width),
            work.color_base_x + travel_width
        )
        : travel_width;
    const auto canvas_height = work.pre_upscale
        ? (std::min)(
            static_cast<std::uint32_t>(color_desc.Height),
            work.color_base_y + travel_height
        )
        : travel_height;
    if (region.base_x + region.width > canvas_width) {
        region.base_x = canvas_width > region.width
            ? (canvas_width - region.width) / 8U * 8U : 0U;
        if (region.base_x + region.width > canvas_width) {
            region.width = canvas_width / 8U * 8U;
        }
    }
    if (region.base_y + region.height > canvas_height) {
        region.base_y = canvas_height > region.height
            ? (canvas_height - region.height) / 8U * 8U : 0U;
        if (region.base_y + region.height > canvas_height) {
            region.height = canvas_height / 8U * 8U;
        }
    }
    if (!view.logged_crop ||
        view.logged_crop_x != region.base_x ||
        view.logged_crop_y != region.base_y ||
        view.logged_crop_w != region.width ||
        view.logged_crop_h != region.height) {
        view.logged_crop = true;
        view.logged_crop_x = region.base_x;
        view.logged_crop_y = region.base_y;
        view.logged_crop_w = region.width;
        view.logged_crop_h = region.height;
        const auto displayed = dlss_nr_displayed_view(
            view_width,
            view_height,
            travel_width,
            travel_height
        );
        trace_event(
            "NR crop view=%llu origin=%u,%u size=%ux%u of %ux%u stereoX=%.3f "
            "source=%.0f,%.0f pre_upscale=%s polish=%s input=%ux%u output=%ux%u "
            "colorTex=%ux%u colorBase=%u,%u outputBase=%u,%u",
            static_cast<unsigned long long>(frame.view_id),
            region.base_x,
            region.base_y,
            region.width,
            region.height,
            displayed.width,
            displayed.height,
            settings.nr_x_offset,
            settings.nr_source_x,
            settings.nr_source_y,
            work.pre_upscale ? "yes" : "no",
            work.after_polish ? "yes" : "no",
            work.input_width,
            work.input_height,
            work.output_width,
            work.output_height,
            static_cast<std::uint32_t>(color_desc.Width),
            static_cast<std::uint32_t>(color_desc.Height),
            work.color_base_x,
            work.color_base_y,
            work.output_base_x,
            work.output_base_y
        );
    }
    // Transport already copied the NR crop into a 0,0-based texture. Do not
    // nest another foveation crop inside that patch (that shifted left).
    if (work.color_is_region) {
        region.base_x = 0U;
        region.base_y = 0U;
        region.width = static_cast<std::uint32_t>(color_desc.Width);
        region.height = color_desc.Height;
        region.shape_width = 1.0F;
        region.shape_height = 1.0F;
    }
    ID3D12Resource* game_color = work.color;
    const auto game_color_state = work.color_state;
    const auto crop_src_x = region.base_x;
    const auto crop_src_y = region.base_y;
    bool copied_input = false;
    const bool color_has_uav =
        (color_desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
    if (work.pre_upscale && !work.color_is_region && !color_has_uav) {
        if (!ensure_input_scratch(
                view, device, region.width, region.height, color_desc.Format
            )) {
            device->Release();
            diagnostics.state = DlssNrState::unsupported_resources;
            ++diagnostics.failed_calls;
            return false;
        }
        const auto local_x = region.base_x > work.color_base_x
            ? region.base_x - work.color_base_x : 0U;
        const auto local_y = region.base_y > work.color_base_y
            ? region.base_y - work.color_base_y : 0U;
        transition(
            work.command_list, game_color, game_color_state,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        );
        transition(
            work.command_list, view.input_scratch,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        copy_texture_region(
            work.command_list, view.input_scratch, 0U, 0U,
            game_color, crop_src_x, crop_src_y, region.width, region.height
        );
        transition(
            work.command_list, view.input_scratch,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        work.color = view.input_scratch;
        work.color_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        work.color_is_region = true;
        work.color_base_x = 0U;
        work.color_base_y = 0U;
        work.depth_base_x += local_x;
        work.depth_base_y += local_y;
        work.depth_width = region.width;
        work.depth_height = region.height;
        work.motion_base_x += local_x;
        work.motion_base_y += local_y;
        work.motion_width = region.width;
        work.motion_height = region.height;
        region.base_x = 0U;
        region.base_y = 0U;
        copied_input = true;
    }
    device->Release();
    const auto working_width = scaled_extent(region.width, nr_settings.nr_working_scale);
    const auto working_height = scaled_extent(region.height, nr_settings.nr_working_scale);
    NrRegion codec_region = region;
    const auto nr_color_desc = work.color->GetDesc();
    if (codec_region.base_x + codec_region.width > nr_color_desc.Width ||
        codec_region.base_y + codec_region.height > nr_color_desc.Height) {
        diagnostics.state = DlssNrState::unsupported_resources;
        ++diagnostics.failed_calls;
        trace_event(
            "NR crop exceeds color texture region=%u,%u %ux%u color=%ux%u "
            "color_is_region=%s",
            codec_region.base_x,
            codec_region.base_y,
            codec_region.width,
            codec_region.height,
            static_cast<std::uint32_t>(nr_color_desc.Width),
            static_cast<std::uint32_t>(nr_color_desc.Height),
            work.color_is_region ? "yes" : "no"
        );
        return false;
    }
    if (!create_feature(view, work, settings, working_width, working_height)) {
        if (copied_input) {
            transition(
                work.command_list, game_color,
                D3D12_RESOURCE_STATE_COPY_SOURCE, game_color_state
            );
        }
        return false;
    }
    auto* const gpu = find_or_create_gpu(
        view,
        work,
        codec_region,
        working_width,
        working_height
    );
    if (gpu == nullptr) {
        if (copied_input) {
            transition(
                work.command_list, game_color,
                D3D12_RESOURCE_STATE_COPY_SOURCE, game_color_state
            );
        }
        return false;
    }
    const auto local_x = region.base_x > eye_base_x ? region.base_x - eye_base_x : 0U;
    const auto local_y = region.base_y > eye_base_y ? region.base_y - eye_base_y : 0U;
    const auto depth_x = work.color_is_region
        ? ScaledSubrect{work.depth_base_x, work.depth_width}
        : scale_subrect(
            local_x,
            region.width,
            work.depth_base_x,
            work.depth_width,
            work.output_width
        );
    const auto depth_y = work.color_is_region
        ? ScaledSubrect{work.depth_base_y, work.depth_height}
        : scale_subrect(
            local_y,
            region.height,
            work.depth_base_y,
            work.depth_height,
            work.output_height
        );
    const auto motion_x = work.color_is_region
        ? ScaledSubrect{work.motion_base_x, work.motion_width}
        : scale_subrect(
            local_x,
            region.width,
            work.motion_base_x,
            work.motion_width,
            work.output_width
        );
    const auto motion_y = work.color_is_region
        ? ScaledSubrect{work.motion_base_y, work.motion_height}
        : scale_subrect(
            local_y,
            region.height,
            work.motion_base_y,
            work.motion_height,
            work.output_height
        );

    auto* const parameters = view.parameters;
    const auto signature = settings_signature(settings, region);
    const auto reset_generation = requested_reset_generation.load(
        std::memory_order_acquire
    );
    const bool reset = work.reset || !view.was_enabled ||
        view.settings_signature != signature ||
        view.reset_generation != reset_generation;
    view.was_enabled = true;
    view.settings_signature = signature;
    view.reset_generation = reset_generation;

    uav_barrier(work.command_list, work.color);
    transition(
        work.command_list,
        work.color,
        work.color_state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    constexpr std::uint32_t hdr_mode = 0U;
    dispatch_codec(
        work,
        *gpu,
        gpu->encode_pipeline,
        0U,
        4U,
        settings,
        codec_region,
        hdr_mode
    );
    uav_barrier(work.command_list, gpu->original_output);
    uav_barrier(work.command_list, gpu->color_proxy);
    transition(
        work.command_list,
        gpu->original_output,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    transition(
        work.command_list,
        gpu->color_proxy,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    parameters->Set("Color", gpu->color_proxy);
    parameters->Set("Output", gpu->neural_output);
    parameters->Set("Depth", work.depth);
    parameters->Set("MotionVectors", work.motion_vectors);
    parameters->Set("DLSSNR.Color", gpu->color_proxy);
    parameters->Set("DLSSNR.Output", gpu->neural_output);
    parameters->Set("DLSSNR.Backbuffer", gpu->color_proxy);
    parameters->Set("DLSSNR.MVec", work.motion_vectors);
    parameters->Set("DLSSNR.Depth", work.depth);
    parameters->Set("DLSSNR.ColorSubrectBaseX", 0U);
    parameters->Set("DLSSNR.ColorSubrectBaseY", 0U);
    parameters->Set("DLSSNR.ColorSubrectWidth", working_width);
    parameters->Set("DLSSNR.ColorSubrectHeight", working_height);
    parameters->Set("DLSSNR.OutputSubrectBaseX", 0U);
    parameters->Set("DLSSNR.OutputSubrectBaseY", 0U);
    parameters->Set("DLSSNR.OutputSubrectWidth", working_width);
    parameters->Set("DLSSNR.OutputSubrectHeight", working_height);
    parameters->Set("DLSSNR.BackbufferSubrectBaseX", 0U);
    parameters->Set("DLSSNR.BackbufferSubrectBaseY", 0U);
    parameters->Set("DLSSNR.BackbufferSubrectWidth", working_width);
    parameters->Set("DLSSNR.BackbufferSubrectHeight", working_height);
    parameters->Set("DLSSNR.DepthSubrectBaseX", depth_x.base);
    parameters->Set("DLSSNR.DepthSubrectBaseY", depth_y.base);
    parameters->Set("DLSSNR.DepthSubrectWidth", depth_x.extent);
    parameters->Set("DLSSNR.DepthSubrectHeight", depth_y.extent);
    parameters->Set("DLSSNR.MVecSubrectBaseX", motion_x.base);
    parameters->Set("DLSSNR.MVecSubrectBaseY", motion_y.base);
    parameters->Set("DLSSNR.MVecSubrectWidth", motion_x.extent);
    parameters->Set("DLSSNR.MVecSubrectHeight", motion_y.extent);
    parameters->Set(
        "DLSSNR.MVecScaleX",
        work.motion_scale_x * settings.nr_motion_scale_x_multiplier *
            static_cast<float>(working_width) / region.width
    );
    parameters->Set(
        "DLSSNR.MVecScaleY",
        work.motion_scale_y * settings.nr_motion_scale_y_multiplier *
            static_cast<float>(working_height) / region.height
    );
    const bool depth_inverted = settings.nr_depth_convention == 1U
        ? false
        : settings.nr_depth_convention == 2U ? true : work.depth_inverted;
    parameters->Set("DLSSNR.DepthInverted", depth_inverted ? 1U : 0U);
    parameters->Set("DLSSNR.Enabled", 1U);
    parameters->Set("DLSSNR.Reset", reset ? 1U : 0U);
    parameters->Set("Jitter.Offset.X", work.jitter_x);
    parameters->Set("Jitter.Offset.Y", work.jitter_y);
    parameters->Set("DLSSNR.JitterX", work.jitter_x);
    parameters->Set("DLSSNR.JitterY", work.jitter_y);
    parameters->Set("DLSSNR.ScalingRatio", 1.0F);
    parameters->Set("DLSSNR.Scale", 1.0F);
    parameters->Set("DLSSNR.Upscaling", 0U);
    parameters->Set("DLSSNR.Intensity", settings.nr_intensity);
    parameters->Set("DLSSNR.LocalToneStrength", settings.nr_local_tone_strength);
    parameters->Set(
        "DLSSNR.LocalStructureStrength",
        settings.nr_local_structure_strength
    );
    parameters->Set(
        "DLSSNR.SkinStructureStrength",
        settings.nr_skin_structure_strength
    );
    parameters->Set("DLSSNR.UseAutoMask", settings.nr_automatic_mask ? 1U : 0U);
    parameters->Set("DLSSNR.Style", 0U);
    parameters->Set("DLSSNR.UICorrection", settings.nr_ui_correction ? 1U : 0U);

    const auto result = runtime.evaluate_feature(
        work.command_list,
        view.handle,
        parameters,
        nullptr
    );
    diagnostics.last_result = result;
    diagnostics.output_width = work.output_width;
    diagnostics.output_height = work.output_height;
    diagnostics.color_width = static_cast<std::uint32_t>(color_desc.Width);
    diagnostics.color_height = static_cast<std::uint32_t>(color_desc.Height);
    const auto displayed_view = dlss_nr_displayed_view(
        view_width,
        view_height,
        travel_width,
        travel_height
    );
    diagnostics.view_width = displayed_view.width;
    diagnostics.view_height = displayed_view.height;
    diagnostics.region_base_x = copied_input ? crop_src_x : region.base_x;
    diagnostics.region_base_y = copied_input ? crop_src_y : region.base_y;
    diagnostics.region_width = region.width;
    diagnostics.region_height = region.height;
    diagnostics.working_width = working_width;
    diagnostics.working_height = working_height;
    diagnostics.intermediate_vram_bytes =
        static_cast<std::uint64_t>(region.width) * region.height * 8U +
        static_cast<std::uint64_t>(working_width) * working_height * 16U;
    if (!ngx_succeeded(result)) {
        restore_original_color(work, *gpu, codec_region);
        if (copied_input) {
            transition(
                work.command_list, game_color,
                D3D12_RESOURCE_STATE_COPY_SOURCE, game_color_state
            );
        }
        diagnostics.state = DlssNrState::evaluation_failed;
        ++diagnostics.failed_calls;
        trace_event(
            "DLSS-NR feature 18 evaluation failed view=%llu result=0x%08X",
            static_cast<unsigned long long>(frame.view_id),
            result
        );
        return false;
    }

    record_color_samples(
        view,
        *gpu,
        work.command_list,
        nr_settings,
        color_desc.Format
    );
    transition(
        work.command_list,
        gpu->neural_output,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    transition(
        work.command_list,
        work.color,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    dispatch_codec(
        work,
        *gpu,
        gpu->decode_pipeline,
        1U,
        6U,
        settings,
        codec_region,
        hdr_mode
    );
    uav_barrier(work.command_list, work.color);
    transition(
        work.command_list,
        work.color,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        work.color_state
    );
    if (copied_input) {
        transition(
            work.command_list, work.color,
            work.color_state, D3D12_RESOURCE_STATE_COPY_SOURCE
        );
        transition(
            work.command_list, game_color,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        copy_texture_region(
            work.command_list, game_color, crop_src_x, crop_src_y,
            work.color, 0U, 0U, region.width, region.height
        );
        transition(
            work.command_list, game_color,
            D3D12_RESOURCE_STATE_COPY_DEST, game_color_state
        );
        transition(
            work.command_list, work.color,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
    }
    transition(
        work.command_list,
        gpu->original_output,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    transition(
        work.command_list,
        gpu->color_proxy,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    transition(
        work.command_list,
        gpu->neural_output,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    ++diagnostics.evaluation_calls;
    diagnostics.state = DlssNrState::active;
    if (diagnostics.evaluation_calls == 1U) {
        trace_event(
            "DLSS-NR active route=%s view=%llu region=%ux%u@%u,%u "
            "working=%ux%u evaluations=%llu pre_upscale=%s polish=%s hdr=%s "
            "flags=0x%X game_flags=0x%X format=0x%X",
            dlss_nr_route_name(work.route),
            static_cast<unsigned long long>(frame.view_id),
            region.width,
            region.height,
            copied_input ? crop_src_x : region.base_x,
            copied_input ? crop_src_y : region.base_y,
            working_width,
            working_height,
            static_cast<unsigned long long>(diagnostics.evaluation_calls),
            work.pre_upscale ? "yes" : "no",
            work.after_polish ? "yes" : "no",
            "reinhard-detail",
            nr_feature_create_flags(work.create_flags),
            work.create_flags,
            static_cast<unsigned>(color_desc.Format)
        );
    }
    return true;
}

void flush_dlss_nr_color_samples() noexcept {
    std::lock_guard lock(nr_mutex);
    for (auto& view : views) log_pending_color_samples(view);
}

void release_dlss_nr_view(const DlssViewId view_id) noexcept {
    std::lock_guard lock(nr_mutex);
    for (auto iterator = views.begin(); iterator != views.end(); ++iterator) {
        if (iterator->view_id != view_id) continue;
        release_feature(*iterator);
        for (auto& gpu : iterator->gpu_resources) release_gpu(gpu);
        views.erase(iterator);
        return;
    }
}

void release_dlss_nr_resources() noexcept {
    std::lock_guard lock(nr_mutex);
    for (auto& view : views) {
        release_feature(view);
        for (auto& gpu : view.gpu_resources) release_gpu(gpu);
    }
    views.clear();
    release(runtime.device);
    release(pending_device);
    runtime_load_started.store(false, std::memory_order_release);
    // Keep the signed module loaded: its shutdown export is not part of the
    // known-good feature-18 contract and unloading it can race queued work.
    runtime = {};
    diagnostics = {};
}

void reset_dlss_nr() noexcept {
    requested_reset_generation.fetch_add(1U, std::memory_order_acq_rel);
    std::lock_guard lock(nr_mutex);
    if (runtime.state == 2U) runtime.state = 0U;
    for (auto& view : views) {
        view.was_enabled = false;
        view.feature_failed = false;
    }
    diagnostics.state = DlssNrState::waiting;
    diagnostics.last_result = 0U;
}

DlssNrSnapshot dlss_nr_snapshot() noexcept {
    std::lock_guard lock(nr_mutex);
    return diagnostics;
}

const char* dlss_nr_state_name(const DlssNrState state) noexcept {
    switch (state) {
    case DlssNrState::waiting: return "Waiting for a compatible DLSS frame";
    case DlssNrState::disabled: return "Disabled in the add-on";
    case DlssNrState::runtime_missing: return "nvngx_dlssnr.dll not found beside the add-on";
    case DlssNrState::runtime_failed: return "DLSS-NR runtime initialization failed";
    case DlssNrState::unsupported_resources: return "Unsupported or missing depth, motion, or output resource";
    case DlssNrState::feature_failed: return "DLSS-NR feature 18 creation failed";
    case DlssNrState::evaluation_failed: return "DLSS-NR feature 18 evaluation failed";
    case DlssNrState::active: return "Active";
    }
    return "Unknown";
}

const char* dlss_nr_route_name(const DlssNrRoute route) noexcept {
    switch (route) {
    case DlssNrRoute::none: return "Waiting";
    case DlssNrRoute::d3d12_native: return "Direct3D 12";
    case DlssNrRoute::d3d11_transport: return "DX11 -> DX12 Transport";
    case DlssNrRoute::streamline: return "Streamline / Direct3D 12";
    }
    return "Unknown";
}

}  // namespace cheeky::foveated_dlss
