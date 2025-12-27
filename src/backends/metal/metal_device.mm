#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device.hpp"

namespace backend
{

class MetalDevice final : public Device
{
private:
  id<MTLDevice> device{MTLCreateSystemDefaultDevice()};

public:
  MetalDevice() = default;

  ~MetalDevice()
  {
    // ARC handles device lifetime
    this->device = nil;
  }

  Type type() const override { return Type::METAL; }
};

std::shared_ptr<Device> make_metal_device() { return std::make_shared<MetalDevice>(); }

} // namespace backend
