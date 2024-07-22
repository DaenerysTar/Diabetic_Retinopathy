<template>
  <div class="home-page">
    <h1>Welcome to Diabetic Retinopathy Diagnostic Platform!</h1>
    <div class="content">
      <div class="module">
        <h2>What is this?</h2>
        <p>
          This is a diagnostic platform for diabetic retinopathy. The platform was created by CAI Yixuan, Mi Tiantian and Zheng Siyi of Tongji University. It is one of the common complications of diabetes, caused by long-term high blood sugar damaging small blood vessels in the eye, and is one of the main causes of vision loss in adults.
        </p>
      </div>
      <div class="module">
        <h2>Our advantage?</h2>
        <p>
          More diverse model selection, more massive training data, more accurate test results, more concise use process.
        </p>
      </div>
    </div>
    <div class="start-button">
      <el-button type="primary" @click="goToPrediction">START</el-button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HomePage',
  methods: {
    goToPrediction() {
      this.$router.push('/prediction');
    }
  }
}
</script>

<style scoped>
.home-page {
  width: 100%;
  height: 100vh;
  background-image: url('@/assets/background.jpg');
  background-size: cover;
  background-position: center;
  display: flex;
  flex-direction: column;
  color: white;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
}

h1 {
  font-size: 40px;
  margin-top: 40px;
  text-align: center;
  margin-bottom: 30px;
}

.content {
  display: flex;
  justify-content: space-around;
  align-items: flex-start;
  width: 100%;
  padding: 20px;
}

.module {
  background: rgba(0, 0, 0, 0.6);
  padding: 20px;
  border-radius: 8px;
  width: 35%;
  max-width: 600px;
}

h2 {
  font-size: 32px;
  margin-bottom: 10px;
}

p {
  font-size: 18px;
}

.start-button {
  display: flex;
  justify-content: center;
  margin-top: 30px;
}

.start-button .el-button {
  background-color: brown;
  height: 40px;
  font-size: 20px;
  padding: 15px 30px;}

</style>
