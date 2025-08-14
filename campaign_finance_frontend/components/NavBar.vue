<template>
  <v-app-bar app color="primary" dark>
    <!-- Logo -->
    <v-toolbar-title class="d-flex align-center">
      <!--<NuxtLink to="/">
        <v-img 
          src="/argus_eye.png" 
          alt="Argus Logo"
          max-height="2000"
          max-width="2000"
          contain
          class="mr-3"
        />
      </NuxtLink>-->
<div class="nav-container">
  <NuxtLink to="/" class="nav-title">ARGUS</NuxtLink>

  <span v-if="whoami?.email" class="text-caption nav-title">
    Welcome,&nbsp;{{ whoami.email }}
  </span>
</div>

    </v-toolbar-title>

    <v-spacer></v-spacer>

    <!-- Navigation Links -->
    <v-btn v-for="link in navLinks" :key="link.to" :to="link.to" text>
      {{ link.label }}
    </v-btn>
  </v-app-bar>
</template>

<script setup>
import { useWhoami } from '@/composables/useWhoami'

const { whoami } = useWhoami()

const navLinks = [
  { label: "Home", to: "/" },
  

  
];
</script>

<style scoped>
.v-toolbar-title {
  display: flex;
  align-items: center;
}

.nav-title {
  color: white;
  font-weight: bold;
  font-size: 1.2rem;
  text-decoration: none;
}

.nav-title:hover {
  text-decoration: underline;
}
.nav-container {
  display: flex;
  flex-direction: column;
}

</style>
