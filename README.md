# neural-network-pipelie
The full pipeline from the quantitative training of the neural network, then conversion to FPGA IPs and the generation of a complete RISC-V SoC.


## Some **IMPORTANT** info:
----
- ### can not find vivado_hls.

For Vi*20, we use vitis_hls, need use vitis branch for finn/finn-base/finn-hlslib to support vitis_hls.

----
- ### step_hls_ipgen can not find project.
```bash
diff --git a/transform/build.py b/transform/build.py
index 0c5da60..dd80523 100644
--- a/transform/build.py
+++ b/transform/build.py
@@ -86,7 +86,7 @@ def select_build_steps(platform):

 # create a release dir, used for finn-examples release packaging
 os.makedirs("release", exist_ok=True)
-os.environ["FINN_BUILD_DIR"]="./build"
+os.environ["FINN_BUILD_DIR"]="/home/wenjun/Code/dios/neural-network-pipeline/transform/finn_build"

```

----
- ### step_hls_ipgen can not find inclue file and gen ip fail.

```bash
--- a/src/finn/custom_op/fpgadataflow/hlscustomop.py
+++ b/src/finn/custom_op/fpgadataflow/hlscustomop.py
@@ -296,12 +296,14 @@ class HLSCustomOp(CustomOp):
         f.close()
         self.code_gen_dict.clear()

+        finn_hlslib_dir = "/home/wenjun/xlnx/finn_work_repo/finn-hlslib"
+        custom_hls_dir = "/home/wenjun/xlnx/finn_work_repo/finn/custom_hls"
         # generate tcl script for ip generation
         self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
         self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
         self.code_gen_dict["$FPGAPART$"] = [fpgapart]
-        self.code_gen_dict["$FINNHLSLIBDIR$"] = ["/workspace/finn-hlslib"]
-        self.code_gen_dict["$FINNHLSCUSTOMDIR$"] = ["/workspace/finn/custom_hls"]
+        self.code_gen_dict["$FINNHLSLIBDIR$"] = [finn_hlslib_dir]
+        self.code_gen_dict["$FINNHLSCUSTOMDIR$"] = [custom_hls_dir]
         self.code_gen_dict["$TOPFXN$"] = [node.name]
         self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]

```

----
- ### some other 'workspace' path need patch, detail in **[finn.patch](finn.patch)** .
----
- ### [ERROR: [BD 5-390] IP definition not found for VLNV](https://github.com/bigzz/neural-network-pipeline/issues/4)