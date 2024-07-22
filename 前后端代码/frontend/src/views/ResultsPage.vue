<template>
  <div class="results-page">
    <h1>Readme Text</h1>
    <iframe class="pdf-viewer" src="/static/sample.pdf"></iframe>
  </div>
</template>

<script>
export default {
  name: 'ResultsPage'
}
</script>

<style scoped>
.results-page {
  display: flex;
  flex-direction: column;
  height: 100vh; /* 视窗高度 */
  margin: 0;
}

h1 {
  margin: 0;
  padding: 1rem;
  background-color: #f5f5f5;
  text-align: center;
}

.pdf-viewer {
  flex: 1;
  width: 100%;
  border: none; /* 去除边框 */
}
</style>
