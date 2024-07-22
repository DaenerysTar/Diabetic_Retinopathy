<template>
  <el-container style="border: 1px solid #eee;height:100vh">
    <el-header style="height:10%; background-color: #1f2d3d;">
      <span style="font-size:24px;color:gold">Diabetic Retinopathy Diagnostic Platform</span>
    </el-header>
    <el-container style="height:90%">
      <el-aside width="180px" style="background-color:#2c3e50;">
        <el-menu :default-openeds="['1']" background-color="#2c3e50" text-color="#b8c7ce" active-text-color="#ffd04b" router>
          <el-submenu index="1">
            <template #title>
              <i class="el-icon-menu"></i><span style="color:gold">菜单</span>
            </template>
            <el-menu-item index="/">Home</el-menu-item>
            <el-menu-item index="/prediction">Online Prediction</el-menu-item>
            <el-menu-item index="/results">About Us</el-menu-item>
          </el-submenu>
        </el-menu>
      </el-aside>
      <el-main style="padding: 20px;">

        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script>
export default {
  name: 'NavBar'
}
</script>

<style scoped>
.el-header {
  background-color: #1f2d3d;
  color: gold;
  line-height: 60px;
  text-align: center;
  font-size: 24px;
  border-bottom: 1px solid #ddd;
}

.header-breadcrumb {
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid #ccc;
  color: #b8c7ce;
}

.el-aside {
  background-color: #2c3e50;
}

.el-menu {
  background-color: #2c3e50;
  color: #b8c7ce;
}

.el-menu-item {
  background-color: #2c3e50;
  color: #b8c7ce;
  transition: background-color 0.3s, color 0.3s;
}

.el-menu-item:hover {
  background-color: #34495e;
  color: #fff;
}

.el-menu-item:active {
  background-color: #1abc9c;
  color: #fff;
}

.el-submenu__title {
  color: gold;
  font-size: 18px;
  font-weight: bold;
}
</style>
