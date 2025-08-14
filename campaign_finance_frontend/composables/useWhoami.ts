// Fetches once per request on the server and once in the browser.
// Nuxt will reuse the same result anywhere else you call it.
import { useAsyncData } from '#app'

export const useWhoami = () => {
  const { data, error } = useAsyncData(
    'whoami',                       // shared key  ➜  one fetch only
    () => $fetch('/api/whoami')
  )
  // data.value looks like { email, username }
  return { whoami: data, whoamiErr: error }
}
