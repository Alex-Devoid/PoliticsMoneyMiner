// plugins/vuetify.ts
/// <reference types="vite/client" />
import { defineNuxtPlugin } from 'nuxt/app'
import { h }                from 'vue'
import {
  createVuetify,
  /* typing helpers for a custom set */
  type IconSet,
  type IconProps,
} from 'vuetify'

import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import 'vuetify/styles'

/* 1️⃣  keep font-based Material Design Icons as the default set */
import '@mdi/font/css/materialdesignicons.css'
import { mdi } from 'vuetify/iconsets/mdi'

/* ------------------------------------------------------------------ */
/* 2️⃣  load every state-outline SVG as a raw string                   */
/* ------------------------------------------------------------------ */
const svgs = import.meta.glob('@/assets/icons/states/*.svg', {
  as: 'raw',
  eager: true,
})

/* convert ".../CA.svg" → { ca:'<svg … />', … }                      */
const stateStrings = Object.fromEntries(
  Object.entries(svgs).map(([path, svg]) => {
    const code = path.split('/').pop()!.replace('.svg', '').toLowerCase()
    return [code, svg as string]
  }),
)

/* ------------------------------------------------------------------ */
/* 3️⃣  custom IconSet called “states”                                 */
/* ------------------------------------------------------------------ */
const statesSet: IconSet = {
  component: (props: IconProps & { class?: any; style?: any }) =>
    h(
      props.tag,
      {
        // forward Vuetify’s sizing / colour classes & inline style
        class : props.class,
        style : props.style,
        /* drop raw SVG markup straight into the wrapper */
        innerHTML: stateStrings[props.icon as string],
      },
    ),
}


/* ------------------------------------------------------------------ */
/* 4️⃣  Vuetify instance                                              */
/* ------------------------------------------------------------------ */
export default defineNuxtPlugin((nuxtApp) => {
  const vuetify = createVuetify({
    components,
    directives,
    theme: { defaultTheme: 'light' },

    icons: {
      /* examples:
         <v-icon icon="mdi:eye"    />
         <v-icon icon="states:ca"  />                                  */
      defaultSet: 'mdi',
      sets: {
        mdi,          // built-in font MDI
        states: statesSet,
      },
    },
  })

  nuxtApp.vueApp.use(vuetify)
})
