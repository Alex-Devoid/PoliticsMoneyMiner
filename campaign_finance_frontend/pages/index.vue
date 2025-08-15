<template>
  <v-app>
    <NavBar />

    <v-main>
      <v-container class="text-center">
        <h1 class="text-h3 font-weight-bold mb-4">Welcome to ARGUS</h1>
        <h1 class="text-subtitle-1 font-weight-thin mb-4">
          Automated Record Gathering and Update System
        </h1>
        <p class="text-h5 mb-6">Watch everything.</p>

        <v-row class="justify-center" dense>
          <v-col cols="12" md="4">
            <NuxtLink to="/filings" prefetch>
              <v-btn color="secondary" class="w-100" size="large">
                Quality-check Filings
              </v-btn>
            </NuxtLink>
          </v-col>

          <v-col cols="12" md="4">
            <NuxtLink to="/finance" prefetch>
              <v-btn color="success" class="w-100" size="large">
                Campaign&nbsp;Finance
              </v-btn>
            </NuxtLink>
          </v-col>

          <!-- Placeholder for future Bills page -->
          <!--
          <v-col cols="12" md="4">
            <NuxtLink to="/bills" prefetch>
              <v-btn color="secondary" class="w-100" size="large">
                Explore Bills
              </v-btn>
            </NuxtLink>
          </v-col>
          -->
        </v-row>
      </v-container>
    </v-main>

    <AppFooter />
  </v-app>
</template>

<script setup>
/* EITHER rely on Nuxt auto-imports (remove both lines),
   OR import with matching names as used in the template: */
import NavBar from "@/components/NavBar.vue";
import AppFooter from "@/components/AppFooter.vue";
</script>
