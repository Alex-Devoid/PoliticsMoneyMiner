import { defineNuxtConfig } from 'nuxt/config'
import vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'

export default defineNuxtConfig({

    
    ssr: true,

    app: {
      head: {
        title: 'ARGUS', // optional default title
        link: [
          { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
          { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' },
          { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' },
          { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }
        ],
        meta: [
          { name: 'description', content: 'Automated Record Gathering and Update System' }
        ]
      }
    },

    // API routes for internal proxying
    nitro: {
      routeRules: {
        "/api/**": { cors: true }, // Ensure API routes support CORS
      },
    },
  
  // Ensure Vuetify works properly in SSR mode
  build: {
    transpile: ['vuetify'],
  },

  // ✅ Register Nuxt Modules (Pinia & Custom Hook for Vuetify)
  modules: [
    '@pinia/nuxt', // 🟢 Auto-register Pinia (No need for `plugins/pinia.ts`)
    (_options, nuxt) => {
      nuxt.hooks.hook('vite:extendConfig', (config) => {
        // Ensure `config.plugins` exists before pushing Vuetify
        config.plugins = config.plugins || []
        config.plugins.push(vuetify({ autoImport: true }))
      })
    },
  ],

  // ✅ Configure Vite for Vuetify & Vue
  vite: {
    vue: {
      template: {
        transformAssetUrls,
      },
    },
  },

  // ✅ Enable Auto Imports (e.g., Pinia stores auto-detected)
  imports: {
    dirs: ['store'], // Auto-imports everything inside `store/`
  },

  // ✅ Runtime Configuration (Use environment variables)\
  // 'https://argus-meetings-api-963969693847.us-central1.run.app',
  runtimeConfig: {
    public: {
      apiUrl: process.env.API_URL,
      financeApiUrl : process.env.FINANCE_API_URL || 'http://backend-campaign-finance-api:8080',
      iapClientId: process.env.IAPClientIdPublic || "",
      devUserEmail : process.env.DEV_USER_EMAIL      || ''
      
    },
  },

  // ✅ TypeScript Configuration
  typescript: {
    strict: true,
    shim: false, // Disable shims for cleaner types
  },
})
