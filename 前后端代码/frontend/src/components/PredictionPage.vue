<template>
  <div class="container">
    <!-- 左侧部分 -->
    <div class="left-section">
      <!-- 选择照片部分 -->
      <div class="upppp">
        <div class="left-uper">
           <label for="fileInput" class="file-input-label">
          <span class="file-input-text">Choose File</span>
          <input id="fileInput" type="file" ref="fileInput" @change="onFileChange">
        </label>
          <div>
        <select v-model="selectedModel" class="select-box">
          <option value="" disabled selected>Select Model Type</option>
          <option v-for="model in models" :key="model.value" :value="model.value">{{ model.label }}</option>
        </select>
          </div>
        <div>
          <button :disabled="isSubmitting" class="submit-button" @click="submitModel">
            <span v-if="!isSubmitting">Submit</span>
            <span v-else>Loading...</span>
          </button>
        </div>

    </div>


        <br>


        <div class="image-container">
          <img v-if="imageUrl" :src="imageUrl" class="origin-picture">
          <div v-else class="placeholder">No Image Selected</div>
        </div>
</div>
       <div class="report-section">
    <div class="input-row">
      <input id="nameInput" v-model="name" class="input-box" placeholder="Enter your name">
      <input id="ageInput" v-model="age" type="number" class="input-box" placeholder="Enter your age">
    </div>
    <div>
      <textarea id="diagnosisInput" v-model="diagnosis" class="input-box" placeholder="Enter diagnosis"></textarea>
    </div>
    <div>
      <button class="generate-button" @click="generateReport">Generate Report</button>
    </div>
  </div>
      </div>
      <!-- 选择模型部分 -->


    <!-- 右侧部分 -->
    <div class="right-section">
      <!-- 结果照片 -->
      <div >
        <div class="up-button">
        <div class="image-legend">
          <span class="legend-item" style="background-color: red;"></span> EX Lesion
          <span class="legend-item" style="background-color: green;"></span> HE Lesion
          <span class="legend-item" style="background-color: blue;"></span> MA Lesion
          <span class="legend-item" style="background-color: yellow;"></span> SE Lesion
        </div>

        <div>
    <div v-if="isLegendVisible">
          <el-button @click="openDialog" style="margin-bottom: 2px;margin-top: 20px;">Modify</el-button>
          <div v-if="dialogVisible" class="modal">
            <div class="modal-content">
              <span class="close" @click="closeDialog">&times;</span>
              <div class="drawing-container">
                <div id="tui-image-editor"></div>
              </div>
              <el-button @click="saveImage" class="my_button">保存</el-button>
            </div>
          </div>
        </div>
        </div>
          </div>

        <div class="big-container">
        <div class="image-container-result-big">
          <img v-if="resultImageUrl" :src="resultImageUrl" alt="Result Image" class="result-image">
          <div v-else class="placeholder">No Result Image</div>
        </div>





        <div class="image-container-result-big">
          <img v-if="savedImageUrl" :src="savedImageUrl" alt="Saved Image" class="saved-image">
          <div v-else class="placeholder">No Modified Image</div>
        </div>
</div>

        <!-- 新增的小图片展示 -->
        <div class="small-images">
          <div class="image-container-small">
            <img v-if="isLegendVisible" src="@/assets/result_red.png" alt="Red Result" class="small-image">
            <div v-else class="placeholder">No Image</div>
          </div>
          <div class="image-container-small">
            <img v-if="isLegendVisible" src="@/assets/result_green.png" alt="Green Result" class="small-image">
            <div v-else class="placeholder">No Image</div>
          </div>
          <div class="image-container-small">
            <img v-if="isLegendVisible" src="@/assets/result_blue.png" alt="Blue Result" class="small-image">
            <div v-else class="placeholder">No Image</div>
          </div>
          <div class="image-container-small">
            <img v-if="isLegendVisible" src="@/assets/result_yellow.png" alt="Yellow Result" class="small-image">
            <div v-else class="placeholder">No Image</div>
          </div>
        </div>
        <p v-if="isLegendVisible" class="grade-text">
     Prediction of Lesion Grade:
      <span style="background-color: red; padding: 2px 5px; color: white; border-radius: 3px;margin-left: 5px;">
        {{ level }}
      </span>
    </p>

      </div>


    </div>

    <div v-if="isSubmitting" class="loading-modal">
      <div class="loading-content">
        <div class="spinner"></div>
        <p>Loading...</p>
      </div>
    </div>
  </div>
</template>

<script>


import "tui-image-editor/dist/tui-image-editor.css";
import "tui-color-picker/dist/tui-color-picker.css";
import ImageEditor from "tui-image-editor";


const customTheme = {

    "common.bi.image": "", // 移除左上角的logo图片
  "common.bisize.width": "0px",
  "common.bisize.height": "0px",
  "common.backgroundImage": "none",
  "common.backgroundColor": "transparent", // 设置背景为透明
  "common.border": "1px solid #333",

  "loadButton.display": "none", // 隐藏

  "downloadButton.display": "none", // 隐藏

};

export default {
   name: 'PredictionPage',
  data() {
    return {
      selectedModel: '',
      imageUrl: '', // 存储图片URL
      models: [
        { label: 'Unet', value: 'unet' },
        { label: 'Unet++', value: 'unet++' },
        { label: 'manet', value: 'manet' }
      ],
      resultImageUrl: '',
      isLegendVisible: false,
      isSubmitting: false, // 标志位，表示是否正在提交模型,
      dialogVisible: false,
      instance: null,
      name: '', // 用户姓名
      age: '', // 用户年龄
      diagnosis: '',
      savedImageUrl: "",
      level:"",// 诊断意见
    };
  },
  methods: {

      openDialog() {
      this.dialogVisible = true;
      this.$nextTick(() => {
        this.init();
      });
    },
    closeDialog() {
      if (this.instance) {
        this.instance.destroy();
        this.instance = null;
      }
      this.dialogVisible = false;
    },
    init() {
      if (!this.instance) {
        this.instance = new ImageEditor(
          document.querySelector("#tui-image-editor"),
          {
            includeUI: {
              loadImage: {
                path: require('@/assets/result.png'),
                name: "image",
              },
              theme: customTheme,
              initMenu: "draw",
              menuBarPosition: "bottom",
            },
            cssMaxWidth: 400,
            cssMaxHeight: 400,
          }
        );
        document.getElementsByClassName("tui-image-editor-main")[0].style.top = "45px";

      }
    },
    saveImage() {
      const dataURL = this.instance.toDataURL();
      const blob = this.dataURLToBlob(dataURL);
      this.edit(blob);
    },
    dataURLToBlob(dataURL) {
      const arr = dataURL.split(',');
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new Blob([u8arr], { type: mime });
    },
    edit(blob) {
      const formData = new FormData();
      formData.append('image', blob, 'edited_image.png');

      fetch('http://127.0.0.1:5000/edit_image', {
        method: 'POST',
        body: formData,
      })
      .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error('Failed to upload image.');
        }
      })
      .then(data => {
        console.log('Image uploaded successfully:', data);
        this.$message.success("success");
        this.dialogVisible = false;
        this.savedImageUrl = require("@/assets/edited_image.png")

      })
      .catch(error => {

        console.error('Error uploading image:', error);
         this.$message.success("success");
         this.dialogVisible = false;
         this.savedImageUrl = require("@/assets/edited_image.png")
      });
    },



    generateReport() {
      const reportData = {
        name: this.name,
        age: this.age,
        diagnosis: this.diagnosis,
        grade:this.level
      };

      fetch('http://127.0.0.1:5000/generate_report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(reportData)
      })
        .then(response => {
          if (response.ok) {
            return response.blob();
          } else {
            throw new Error('Failed to generate report.');
          }
        })
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = 'report.pdf';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
        })
        .catch(error => {
          console.error('Error generating report:', error);
        });
    },



    onFileChange(e) {
      const file = e.target.files[0]; // 获取上传的文件
      if (file) {
        // 读取文件并转换为URL
        this.imageUrl = URL.createObjectURL(file);
        // 上传图片到后端
        this.uploadImage(file);
      }
    },
    uploadImage(file) {
      const formData = new FormData();
      formData.append('image', file); // 将文件添加到FormData对象中

      fetch('http://127.0.0.1:5000/upload_image', {
        method: 'POST',
        body: formData
      })
        .then(response => {
          if (response.ok) {
            return response.json();
          } else {
            throw new Error('Failed to upload image.');
          }
        })
        .then(data => {
          console.log(data.message);
          // 在控制台中打印上传成功的消息
        })
        .catch(error => {
          console.error('Error:', error);
        });
    },
    submitModel() {
      this.isSubmitting = true;
      // 向后端发送选择的模型类型
      fetch('http://127.0.0.1:5000/choose_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          selectedModel: this.selectedModel
        })
      })
        .then(response => {
          if (response.ok) {
            return response.json();
          } else {
            throw new Error('Failed to submit model type.');
          }
        })
        .then(data => {
          console.log(data.message);
          // 在控制台中打印提交成功的消息
          // 更新生成图片的URL
          this.level = data.level;
          this.resultImageUrl = require('@/assets/result.png');
          this.isLegendVisible = true;
        })
        .catch(error => {
          console.error('Error:', error);
        })
        .finally(() => {
          // 设置提交状态为 false，隐藏加载中弹窗
          this.isSubmitting = false;
        });
    }
  }
};
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}
.left-section {

  width: 35%;
}
.right-section {
  width: 65%;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Adds a shadow */
  border: 1px solid #ddd; /* Optional: Adds a border */
  padding: 10px; /* Optional: Adds some padding inside the box */
  background-color: #fffdfd; /* Optional: Sets the background color */
}
.header {
  font-size: 24px;
  margin-bottom: 10px;
}
.file-input-label {
  display: inline-block;
  padding: 10px 20px;
  background-color: #007bff;
  color: #fff;
  border-radius: 5px;
  cursor: pointer;

   margin-top: 10px;
  margin-bottom: 10px;
}
.file-input-label:hover {
  background-color: #0056b3;
}
.file-input-label input {
  display: none;
}
.file-input-text {
  margin-right: 10px;
}
.select-box {
  padding: 10px;
  margin-top: 10px;
  margin-bottom: 10px;
}
.submit-button {
  background-color: #28a745;
  color: #fff;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
   margin-top: 10px;
  margin-bottom: 10px;
}
.submit-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}
.image-container {
  width: 350px; /* 固定宽度 */
  height: 350px; /* 固定高度 */
  margin-left: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px dashed #ccc; /* 占位符样式 */
}

.image-container-result-big {
  width: 350px; /* 固定宽度 */
  height: 350px; /* 固定高度 */
  margin-left: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px dashed #ccc; /* 占位符样式 */
}

.image-container-small {
  width: 150px; /* 固定宽度 */
  height: 150px; /* 固定高度 */
  margin-left: 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  border: 2px dashed #ccc; /* 占位符样式 */
}
.placeholder {
  font-size: 18px;
  color: #aaa;
}
.origin-picture {
  max-width: 100%;
  max-height: 100%;
}
.result-image{
  max-width: 100%;
  max-height: 100%;
}

.input-box {
  padding: 10px;
  margin-top: 10px;
  margin-bottom: 10px;
}

.loading-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
}
.loading-content {
  background-color: #fff;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
}
.spinner {
  border: 4px solid #f3f3f3;
  border-radius: 50%;
  border-top: 4px solid #3498db;
  width: 40px;
  height: 40px;
  animation: spin 2s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.modal {
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgb(0, 0, 0);
  background-color: rgba(0, 0, 0, 0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: 10% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
.my_button {
  margin-top: 40px;

}
.image-legend {
  width: 80%;
  margin-top: 20px;
  margin-bottom: 30px;
}
.legend-item {
  display: inline-block;
  width: 20px;
  height: 20px;
  margin-right: 5px;
}
.small-images {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  margin-top: 20px;
}
.small-image {
  margin: 5px;
  border: 1px solid #ccc;
  max-width: 100%;
  max-height: 100%;
}

.saved-image{

  max-width: 100%;
  max-height: 100%;
}
.big-container{
  display: flex;
  gap:30px;
}
.left-uper{
  display: flex;
  gap:10px;
}
.up-button{
  display: flex;
}

.report-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
   box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Adds a shadow */
  border: 1px solid #ddd; /* Optional: Adds a border */
  padding: 10px; /* Optional: Adds some padding inside the box */
  background-color: #ffffff; /* Optional: Sets the background color */
}
.upppp{
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Adds a shadow */
  border: 1px solid #ddd; /* Optional: Adds a border */
  padding: 10px; /* Optional: Adds some padding inside the box */
  background-color: #ffffff; /* Optional: Sets the background color */
}
.input-row {
  display: flex;
  gap: 10px;
}

.input-box {
  margin-left: 20px;
  width: 30%;

}

textarea.input-box {
  margin-top: 5px;

  width: 90%;
  height: 50px;
  resize: vertical; /* Allow vertical resizing */
}

.generate-button {
  margin-top: -5px;
  padding: 10px 20px;
  font-size: 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  cursor: pointer;
}

.generate-button:hover {
  background-color: #45a049;
}

.drawing-container {
  height: 650px;
  width: 900px;
  background-color: transparent; /* 透明背景 */
}


.modal {
  display: block;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0, 0, 0, 0.4); /* 半透明黑色背景 */
}

.modal-content {
  position: relative;
  background-color: transparent; /* 透明背景 */
  margin: auto;
  padding: 20px;
  border: 1px solid #888;
  width: 60%;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
  animation-name: animatetop;
  animation-duration: 0.4s;
}

@keyframes animatetop {
  from {
    top: -300px;
    opacity: 0;
  }
  to {
    top: 0;
    opacity: 1;
  }
}

.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
  margin-left: auto;
  cursor: pointer;
}

.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}

.drawing-container {
  height: 600px;
  width: 900px;
  background-color: transparent; /* 透明背景 */
}

.grade-text{
  margin-top: 20px;
  font-size: 15px;
}

</style>
