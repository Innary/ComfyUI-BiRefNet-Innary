

# ComfyUI-BiRefNet-Innary

Better BiRefNet version for [ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO) in ComfyUI 



## 项目介绍 | Info

- 简单将 [ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO)中的模型替换成[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)的最新版模型，精度有所提升。
  
- 使用方法和[ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO)基本相同。

## 安装 | Install

- 所需依赖：timm，如已安装无需运行 requirements.txt，只需 git 项目即可

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO.git`
    3. `cd custom_nodes/ComfyUI-BiRefNet-ZHO`
    4. `pip install -r requirements.txt`
    5. 重启 ComfyUI


## 使用说明 | How to Use

- 直接使用即可，不需要特殊配置。
- `BiRefNet Model Loader`：从huggingface加载 [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 模型
- `BiRefNet`：使用 [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 模型进行推理
- **具体工作流请参考[sample-workflow](https://github.com/Innary/ComfyUI-BiRefNet-Innary/tree/main/sample_workflow)**


## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=-Innary/ComfyUI-BiRefNet-Innary&type=Date)](https://star-history.com/#Innary/ComfyUI-BiRefNet-Innary&Date)


## Credits

[BiRefNet](https://github.com/zhengpeng7/birefnet)

代码参考了 [viperyl/ComfyUI-BiRefNet](https://github.com/viperyl/ComfyUI-BiRefNet) 和 [ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO)感谢！
