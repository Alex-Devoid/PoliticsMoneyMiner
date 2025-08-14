<template>
  <v-app>
    <NavBar />

    <v-main>
      <v-container class="text-center">
        <h1 class="text-h3 font-weight-bold mb-4">Welcome to ARGUS</h1>
        <h1 class="text-subtitle-1 font-weight-thin mb-4">Automated Record Gathering and Update System</h1>
        <p class="text-h5 mb-6">Watch everything.</p>

          
          <v-col cols="12" md="4">
            <NuxtLink to="/filings" prefetch>
              <v-btn color="secondary" block large>Quality-check Filings</v-btn>
            </NuxtLink>
          </v-col>

          <v-col cols="12" md="4">
            <NuxtLink to="/finance" prefetch>
              <v-btn color="success" block large>Campaign&nbsp;Finance</v-btn>
            </NuxtLink>
          </v-col>          

          <!-- existing placeholder for future Bills page (unchanged) -->
          <!--
          <v-col cols="12" md="4">
            <NuxtLink to="/bills">
              <v-btn color="secondary" block large>Explore Bills</v-btn>
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
import NavBar from "@/components/NavBar.vue";
import Footer from "@/components/AppFooter.vue";
</script>
