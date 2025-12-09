export async function submitPipeline({ file, targetColumn, context, runTraining, apiBase }) {
  const form = new FormData();
  form.append('dataset', file);
  if (targetColumn) form.append('targetColumn', targetColumn);
  if (context) form.append('context', context);
  form.append('runTraining', runTraining);

  const endpoint = `${apiBase || ''}/api/analyze`;
  const res = await fetch(endpoint, {
    method: 'POST',
    body: form,
  });

  const data = await res.json();
  if (!res.ok) {
    const error = data?.error || 'Pipeline run failed';
    const err = new Error(error);
    err.response = data;
    throw err;
  }

  return data;
}
