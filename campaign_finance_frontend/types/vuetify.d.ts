// types/vuetify.d.ts
import 'vuetify'                       // patches the module

declare module 'vuetify' {
  /** allow arbitrary `state-xx` aliases */
  interface IconAliases {
    [key: `state-${string}`]: string | import('vue').Component
  }
}
